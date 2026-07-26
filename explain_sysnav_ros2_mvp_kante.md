# sysnav_ros2_mvp 구조 설명

경로: `ai_module/src/sysnav_ros2_mvp/`. 하나의 ROS2 노드(`sysnav_node`)가 "질문 수신 →
탐지/3D grounding → object 후보 선택 → 목표 접근 or 탐색"을 상태 머신으로 돌리는 단일-룸(Room)
MVP 패키지다. 실행/설치 방법은 `readme_kante2.md`, 토픽/파이프라인 개요는
`ai_module/src/sysnav_ros2_mvp/README.md`, Viewpoint coverage 수식은 같은 폴더의
`VIEWPOINT_IMPLEMENTATION.md` 참고. 이 문서는 **코드 구조 자체**를 폴더/파일 단위로 설명한다.

## 1. 디렉토리 구조

```text
sysnav_ros2_mvp/
├── launch/sysnav.launch.py        ros2 launch 진입점 (Node 하나만 기동)
├── setup.py / setup.cfg / package.xml   colcon 빌드 설정
└── sysnav/
    ├── main.py                     rclpy.init → SysNavNode → MultiThreadedExecutor(4 threads)
    ├── config.py                   전역 상수 + 환경변수 (토픽명, YOLO/SAM2/Gemini, TF, 임계값 전부)
    ├── sysnav_node.py               SysNavNode — 상태 머신 + worker 오케스트레이션 (핵심 파일)
    ├── ros_helpers.py               ROS msg ↔ numpy 변환, timestamp 동기화 유틸
    │
    ├── task/
    │   └── query_parser.py          질문 문장 → {target, attributes, relation, reference_objects, detection_prompts}
    │
    ├── perception/                  프레임 단위 2D→3D 인식 파이프라인
    │   ├── perception_pipeline.py    PerceptionPipeline.process() = detector→segmenter→grounder→debug저장
    │   ├── detector.py                YoloWorldDetector (open-vocab 2D bbox)
    │   ├── segmenter.py               Sam2Segmenter (bbox-prompt mask)
    │   ├── lidar_grounding.py         PanoramaLidarGrounder (mask 안 LiDAR → 3D position/extent/point cloud)
    │   └── debug_visualize.py         save_debug_image() → ai_module/debug/sysnav_detect_*.jpg
    │
    ├── memory/                      프레임 간 지속되는 object 저장소
    │   ├── object_memory.py          ObjectMemory — observation을 새 노드 생성 or 기존 노드 merge
    │   └── object_association.py     category+distance+shape+appearance 유사도 스코어링
    │
    ├── scene_graph/                 Room–Viewpoint–Object 구조화 그래프 (논문 스타일)
    │   ├── scene_graph_manager.py     SceneGraphManager — viewpoint 생성 판단, edge upsert, on-demand relation 추론 호출, export
    │   ├── viewpoint_coverage.py      ViewpointCoverageBuilder — LiDAR ray → voxel coverage set C(v)
    │   └── scene_graph_visualizer.py  snapshot → JSON/DOT/PNG 파일로 export
    │
    ├── reasoning/                   의사결정/추론
    │   ├── gemini_selector.py         GeminiSelector — 후보 중 질문에 맞는 1개 선택 (Gemini → confidence/거리 fallback)
    │   └── spatial_relation_reasoner.py  SpatialRelationReasoner — object-object 관계 검증 (Gemini vision → 3D geometry fallback)
    │
    ├── exploration/                 미탐색 영역 계획
    │   ├── coverage_planner.py        CoveragePlanner — LiDAR ray-tracing occupancy grid + A* + frontier 정렬
    │   ├── frontier_extractor.py      FrontierExtractor — free/unknown 경계 클러스터 추출
    │   └── viewpoint_memory.py        ViewpointMemory — 방문한 exploration goal 기록(중복 방지). scene_graph의 Viewpoint와는 별개 개념
    │
    └── navigation/
        └── goal_publisher.py          GoalPublisher — Pose2D 발행 + object standoff pose 계산
```

`tests/`에는 위 모듈들에 대응하는 단위 테스트(`test_association.py`, `test_frontier.py`,
`test_query_parser.py`, `test_scene_graph.py`, `test_viewpoint_coverage.py`, `test_gemini_works.py`)
가 있다.

## 2. 전체 데이터 흐름

```text
/challenge_question
        ↓ query_parser.extract_target()
{target, attributes, relation, reference_objects, detection_prompts}
        ↓
┌─────────────────────────── perception_job (worker thread) ───────────────────────────┐
│ image_rgb + points_sensor + prompts + robot_pose                                     │
│   → PerceptionPipeline.process()                                                     │
│        YoloWorldDetector.detect()   → 2D bbox                                        │
│        Sam2Segmenter.segment()      → bbox 안 mask                                    │
│        PanoramaLidarGrounder.ground() → mask 안 LiDAR → 3D position/extent/point cloud │
│   → ObjectMemory.update()            → 신규/병합 object_id 목록                        │
│   → SceneGraphManager.add_observation()                                              │
│        - object 노드 sync + room_object edge                                          │
│        - LiDAR coverage 새로울 때만 Viewpoint 노드 생성 (아래 §5)                        │
│        - task에 relation 있으면 공통 Viewpoint 검색 → object_object edge 추론(§6)       │
└────────────────────────────────────────────────────────────────────────────────────┘
        ↓ candidates(= target category object들) 존재?
   있음 → SELECT_TARGET                       없음 → PLAN_EXPLORATION
        ↓ selection_job                             ↓ exploration_job
   ObjectMemory.find_by_category()             CoveragePlanner.plan_route()
   → SceneGraphManager.find_matching_target_ids()  (occupancy grid + frontier + A*)
     (relation 검증된 edge 있으면 후보 필터링)         ↓
   → GeminiSelector.select()                   FOLLOW_EXPLORATION (waypoint 순서대로 이동)
   → GoalPublisher.object_approach_pose()            ↓ 이동 중 주기적으로 perception 재실행
        ↓                                            ↓ 후보 발견 시 SELECT_TARGET로 전환
   NAVIGATE_TARGET → (거리 도달) → SUCCESS      goal 소진 시 OBSERVE로 복귀
        ↓
/way_point_with_heading  (geometry_msgs/Pose2D)
```

## 3. `sysnav_node.py` — 상태 머신과 스레드 구조

`SysNavNode`는 ROS 콜백과 무거운 연산(YOLO/SAM2/Gemini/occupancy update)을 분리한다.

- **ROS 콜백(메인 스레드, `ReentrantCallbackGroup`)**: `question_callback`, `state_callback`,
  `image_callback`, `scan_callback` — 전부 **메시지를 버퍼/변수에 저장만** 하고 즉시 리턴한다.
  실제 연산은 하지 않는다.
- **두 개의 독립된 `ThreadPoolExecutor(max_workers=1)`**
  - `self.worker` — 상태 머신이 매 tick(`control_loop`, 0.2초 주기)마다 필요하면 하나씩
    맡기는 `perception_job` / `selection_job` / `exploration_job`. `active_future`가 있으면
    새 작업을 넣지 않아 **한 번에 하나만 실행**된다.
  - `self.map_worker` — `scan_callback`에서 0.35초(`MAP_UPDATE_INTERVAL_SEC`)마다 독립적으로
    제출하는 `mapping_job`(occupancy grid 갱신). 상태 머신과 무관하게 항상 돌아간다.
- **상태 머신** (`control_loop` → `consume_future` → 상태 분기):

```text
IDLE
  ↓ question_callback (질문 수신)
OBSERVE ──(candidates 있음)──────────────────→ SELECT_TARGET ──→ NAVIGATE_TARGET ──→ SUCCESS
  ↑                                                  │(실패)
  │                                                  ↓
  │                                          PLAN_EXPLORATION
  │(candidates 없음)                                 ↓
  └────────────────────────────────────────  FOLLOW_EXPLORATION
                                              │  ↑ (goal 도달, 다음 waypoint)
                                              │  └ 소진 시 OBSERVE로 복귀
                                              └ 이동 중에도 주기적으로 perception 재실행
                                                (후보 나오면 즉시 SELECT_TARGET)
FAILED  (exploration_job이 route를 못 찾으면 여기서 멈춤 — 복구 로직 없음)
```

- `consume_future()`는 작업 종류(`kind`)별로 실패 시 복구 상태를 다르게 정한다
  (`perception` 실패 → 기원 상태에 따라 `PLAN_EXPLORATION`/`FOLLOW_EXPLORATION` 유지,
  `selection` 실패 → `PLAN_EXPLORATION`, `exploration` 실패 → `FAILED`).
- `expected_task_id != self.task_id` 체크로, 비동기 작업 도중 새 질문이 들어오면 이전
  결과를 버리는 안전장치가 있다.

## 4. `perception/` — 2D→3D 인식 파이프라인

`PerceptionPipeline.process()`가 세 어댑터를 순서대로 호출한다 (`perception_pipeline.py:19`).

1. `YoloWorldDetector.detect()` — open-vocabulary YOLOv8x-WorldV2. 매 호출 `detection_prompts`가
   바뀌면 `set_classes()`로 클래스 목록을 다시 세팅한다.
2. `Sam2Segmenter.segment()` — YOLO bbox를 SAM2의 box prompt로 사용해 mask만 뽑는다
   (`multimask_output=False`). `SAM2_MIN_MASK_AREA_PX` 미만이면 버림.
3. `PanoramaLidarGrounder.ground()` — LiDAR point를 이미지에 투영(`_project`, equirectangular
   yaw/pitch 매핑) → 각 SAM2 mask 안에 들어오는 point만 선택 → sensor→base→map 좌표 변환 →
   5~95 percentile로 robust min/max, position(median), extent 계산.
4. `debug_visualize.save_debug_image()` — bbox+mask+3D position을 오버레이해 `DEBUG_DIR`에 저장.

세 어댑터 모두 지연 로딩(`_load()`)이라 노드 기동 직후엔 모델이 없고, 첫 호출 시 로드된다.

## 5. `memory/` — Object Memory (프레임 간 병합)

`ObjectMemory.update(observations)`가 매 perception_job마다 호출된다.

- 같은 category의 기존 노드들과 `object_association.find_best_match()`로 비교한다.
  스코어 = `WEIGHT_DISTANCE·distance_score + WEIGHT_SHAPE·shape_score + WEIGHT_APPEARANCE·appearance_score`
  (거리는 가우시안 감쇠, 크기는 extent 상대오차, 외형은 HSV 히스토그램 코사인 유사도).
- 임계값(`ASSOCIATION_THRESHOLD`) 이상이면 기존 노드에 `_merge()` (위치는 관측 횟수 기반
  지수이동평균, point cloud는 concat 후 최대 4096개로 서브샘플), 아니면 `_new_node()`로
  새 object_id 발급.
- `find_by_category()` / `get()` / `all_nodes()`는 전부 **deep-copy**를 반환해 외부에서
  내부 상태를 실수로 변형하지 못하게 한다.

## 6. `scene_graph/` + `reasoning/` — Room·Viewpoint·Object 그래프

논문(SysNav) 스타일 구조화 scene graph. 현재는 Single Room이라 `Room_0` 하나만 있다.

**Viewpoint는 매 프레임 생성하지 않는다** (`scene_graph_manager.py:add_observation`):

```text
C_t    = 현재 pose에서 360° LiDAR로 관측된 map-frame voxel 집합  (viewpoint_coverage.py)
C_prev = 지금까지 저장된 모든 Viewpoint coverage의 합집합
C_novel = C_t - C_prev
|C_novel| > VIEWPOINT_NOVEL_VOXEL_THRESHOLD(=omega) 인 경우에만
    → 새 Viewpoint 노드 생성 + panorama 이미지 저장 + room_viewpoint/viewpoint_object edge 생성
```

**Object-Object edge는 사전에 전부 만들지 않고 on-demand로 생성한다**
(`_infer_task_relations_from_common_viewpoints`, `reasoning/spatial_relation_reasoner.py`):

```text
질문에 relation(on/near/left_of/between/...)이 있을 때만
  → target·reference object를 모두 observes하는 기존 Viewpoint들을 검색
  → 저장된 panorama 이미지를 다시 로드
  → Gemini vision으로 관계 검증 (annotated image + bbox 목록)
  → 실패/미설정 시 해당 Viewpoint pose 기준 3D bbox 기하 판정으로 fallback
     (near=xy거리, left/right/front/behind=viewpoint-local 좌표축, on/above/under=수직 gap+수평 포함,
      between=선분 투영)
  → 성립하면 object_object edge 생성, `selection_job`이 이 edge의 source object를 후보 우선순위로 사용
```

- `scene_graph_visualizer.py`는 그래프 스냅샷을 매 갱신마다 `DEBUG_DIR`의
  `scene_graph_latest.json`/`.dot`/`.png`로 **덮어쓴다** (atomic write, `os.replace`).
- 관계 재검증 중복 방지를 위해 `(task_signature, viewpoint_id)` 조합을 `_relation_checks`
  집합에 기록해 같은 조합은 다시 안 본다.

## 7. `reasoning/gemini_selector.py` — 목표 후보 최종 선택

`selection_job`이 후보(같은 category, 필요시 relation으로 필터링된)를 넘기면
`GeminiSelector.select()`가 질문 원문 + 각 후보의 대표 crop 이미지 + 3D 요약(position/extent/
confidence/observation_count)을 Gemini에 전달해 object_id 하나를 고른다. 후보가 1개면 API
호출 없이 바로 반환. Gemini 실패/미설정 시 `_fallback()`이 `confidence - 0.02*거리`로 대체
선택한다.

## 8. `exploration/` + `navigation/` — 목표 후보가 없을 때

- `CoveragePlanner`는 `scan_callback`이 주기적으로 넣어주는 LiDAR로 고정 크기
  (`MAP_SIZE_M`) occupancy grid를 ray-tracing(Bresenham)으로 갱신한다(`update_from_scan`,
  `map_worker`에서 실행 — 상태 머신과 별개 스레드).
- `plan_route()`: occupancy를 clearance만큼 dilate → `FrontierExtractor.extract()`로
  free/unknown 경계 클러스터 추출 → 각 frontier까지 A*(`_astar_length`) 경로 길이 계산 →
  `coverage + cluster_weight·cluster_size - distance_weight·거리` 스코어로 상위 K개
  (`FRONTIER_TOP_K`) 선택 → 로봇 현재 위치에서 가까운 순으로 그리디 정렬(`_order`).
  `ViewpointMemory.is_near_visited()`로 이미 방문한 곳 근처는 제외한다.
- `GoalPublisher.publish()`가 `/way_point_with_heading`(`Pose2D`)로 발행. 목표 object
  접근 시엔 `object_approach_pose()`가 `TARGET_STANDOFF_DISTANCE_M`만큼 떨어진 지점 +
  물체를 향한 yaw를 계산한다.
- 도착 판정은 별도 navigation feedback이 없어 **odometry 거리**로만 판단
  (`goal_reached`, `GOAL_REACHED_DISTANCE_M`).

## 9. `task/query_parser.py` — 질문 파싱

정규식 기반 경량 파서. `_LEADING_COMMAND`로 "Find/locate/..." 같은 명령어를 떼고,
`_RELATIONS` 목록으로 `in front of`/`left of`/`between` 등 관계어를 찾아 target/reference
구절을 분리한 뒤, `_ATTRIBUTES`(색상/재질/크기 등)와 명사를 나누고 `_singularize()`로 복수형을
단수화한다. 최종적으로 `target`, `attributes`, `relation`(canonical 이름), `reference_objects`,
`detection_prompts`(YOLO-World에 넘길 유니크 프롬프트 목록)를 반환한다. LLM 호출 없이 순수
규칙 기반이라 빠르지만, `_RELATIONS`/`_ATTRIBUTES`에 없는 표현은 인식 못 한다.

## 10. `config.py` — 한눈에 보는 튜닝 포인트

| 영역 | 주요 값 |
|---|---|
| 토픽 | `TOPIC_QUESTION/STATE/IMAGE/SCAN/WAYPOINT` |
| 타이밍 | `CONTROL_PERIOD_SEC`(0.2), `SENSOR_SYNC_TOLERANCE_SEC`(0.3), `MAP_UPDATE_INTERVAL_SEC`(0.35) |
| 모델 | `YOLO_WORLD_WEIGHTS/CONFIDENCE/IOU`, `SAM2_CHECKPOINT/DEVICE`, `GEMINI_MODEL` |
| TF (실측 필요) | `T_LIDAR_TO_CAMERA`, `T_SENSOR_TO_BASE`, `PANORAMA_YAW/PITCH_OFFSET_DEG` |
| Object Association | `ASSOCIATION_MAX_DISTANCE_M`, `ASSOCIATION_THRESHOLD`, `ASSOCIATION_WEIGHT_*` |
| Occupancy/Frontier | `MAP_RESOLUTION_M`, `ROBOT_CLEARANCE_M`, `FRONTIER_TOP_K`, `FRONTIER_*_WEIGHT` |
| Scene Graph | `SCENE_GRAPH_SINGLE_ROOM_*`, `SCENE_GRAPH_RELATION_MIN_CONFIDENCE`, `SCENE_GRAPH_*_TOLERANCE_M`(기하 fallback) |
| Viewpoint coverage | `VIEWPOINT_COVERAGE_DISTANCE_M`(d_cover), `VIEWPOINT_COVERAGE_VOXEL_SIZE_M`, `VIEWPOINT_NOVEL_VOXEL_THRESHOLD`(omega) |

대부분 `os.getenv(..., 기본값)`이라 `.env`로 덮어쓸 수 있다 (실제 값은 `readme_kante2.md` 참고).

## 11. 알려진 한계 (README.md "미포함 범위" 요약)

Multi-Room 분할/Room 자동분류/Cross-Room navigation 없음, 정식 SLAM/loop closure 없음,
`/state_estimation`을 전역 map pose로 그대로 신뢰함, navigation 도착 판정에 action feedback이
없어 odometry 거리로만 근사, `FAILED` 상태에서 자동 복구 로직 없음.
