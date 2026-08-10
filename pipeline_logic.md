# sysnav 파이프라인 로직 정리 (인풋 → 아웃풋)

`ai_module/src/sysnav_ros2_mvp/sysnav/` 전체 흐름을 입력부터 출력까지 순서대로 정리한 문서.
"어디서 뭘 하는지" 찾을 때 참고용. 미션별 요구사항/논문 근거는 `MISSION_1/2/3_*_CLAUDE.txt`,
실행 명령어는 `readme_kante2.md` 참고.

## 0. 전체 그림

```
/challenge_question (String)
        │
        ▼
question_callback ──▶ mission_classifier로 미션 타입 결정 ──▶ 미션별 파서로 파싱
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  공용 상태머신 (control_loop, 0.2초 주기)                      │
│  OBSERVE → PLAN_EXPLORATION → FOLLOW_EXPLORATION → (반복)     │
│           └─ 빈 route면 cross-room 시도 후 미션별 종료 처리로   │
└─────────────────────────────────────────────────────────────┘
        │
        ▼ (미션마다 갈라짐, missions/mission{1,2,3}_pipe.py)
┌───────────────┬───────────────────┬─────────────────────────┐
│ Mission 1      │ Mission 2         │ Mission 3               │
│ Numerical      │ Object Reference  │ Instruction-Following   │
│ 탐색 소진→카운트│ SELECT→NAVIGATE   │ step 순회 SELECT↔NAVIGATE│
└───────────────┴───────────────────┴─────────────────────────┘
        │                  │                    │
        ▼                  ▼                    ▼
/numerical_response  /selected_object_marker +  /way_point_with_heading
   (Int32)              /way_point_with_heading    (Pose2D 시퀀스)
```

디버깅용 실시간 대시보드: `ai_module/debug/mission_status_latest.html`
(`docker/ui_checker.sh`로 열기, `mission_dashboard.py`가 매 사이클 갱신).

---

## 1. 입력 — 질문이 들어오는 순간 (`sysnav_node.py::question_callback`)

1. `/challenge_question`(String)으로 문장이 들어옴.
2. **미션 분류** (`task/mission_classifier.py::classify_mission`) — 정규식 기반:
   - `^(how many|count)` → **Numerical**
   - `go (to|near|between)`, `take the path`, `avoid(ing) the path`, `stop (at|by)`, `pass by`, `first,` 등 이동 동사가 있으면 → **Instruction-Following**
   - 나머지(명사구, "Find the X"/"The X...") → **Object Reference**
   - `questions.json` 실제 문장 75개 전수 검증, 100% 정확.
3. **파싱**(미션에 따라 분기):
   - Numerical / Object Reference → `task/llm_query_parser.py::LLMQueryParser.parse()` — Gemini로 `G=(target, attributes, relation_chain)` 추출, 실패 시 `task/query_parser.py::extract_target()`(정규식)로 자동 폴백.
   - Instruction-Following → `missions/mission3_pipe.py::parse_instruction()` (아래 8장 참고).
4. `task["mission_type"]` 필드에 분류 결과 저장, `self.task`에 보관, `state="OBSERVE"`로 리셋.
5. `config.KEEP_MEMORY_BETWEEN_TASKS`(기본 True)면 이전 질문에서 쌓은 `object_memory`/`scene_graph`/`coverage_planner` 맵을 그대로 유지 (같은 씬에서 여러 질문을 순서대로 받는 대회 구조 가정).

---

## 2. 공용 상태머신 (`sysnav_node.py::control_loop` / `consume_future`)

ROS 타이머(`config.CONTROL_PERIOD_SEC=0.2초`)마다 `control_loop()` 호출. 무거운 작업(YOLO/SAM2/Gemini/A*)은 전부 `ThreadPoolExecutor`(`self.worker`, `self.map_worker`)로 던지고, `consume_future()`가 완료된 결과를 받아 상태를 갱신하는 구조.

### 공용 state (세 미션이 그대로 공유)
- `OBSERVE`: 최신 이미지+LiDAR로 `perception_job` 제출.
- `PLAN_EXPLORATION`: `exploration_job`(=`coverage_planner.plan_route()`) 제출.
- `FOLLOW_EXPLORATION`: 계획된 waypoint 시퀀스를 하나씩 `goal_publisher.publish()`, 도착하면 다음 것.

### 미션별 state (각 `missions/mission{1,2,3}_pipe.py`가 처리)
| 미션 | 전용 state |
|---|---|
| 1 (Numerical) | `MISSION1_FINALIZE_COUNT` |
| 2 (Object Reference) | `SELECT_TARGET`, `NAVIGATE_TARGET` |
| 3 (Instruction-Following) | `MISSION3_SELECT_STEP`, `MISSION3_NAVIGATE_STEP` |

`control_loop`은 공용 state까지는 직접 처리하고, 그 외 state는 `task["mission_type"]`으로 해당 `missionN_pipe.loop()`에 위임(`sysnav_node.py`의 `_MISSION_PIPES` dict). `consume_future`도 마찬가지로 job 결과를 `missionN_pipe.on_job_result()`에 위임 — 단 `perception` 결과의 공용 부분(스탬프 갱신, scene graph 로그, marker publish)은 위임 전에 먼저 처리.

### PLAN_EXPLORATION이 빈 route를 반환했을 때 (`consume_future`)
바로 미션별 종료 처리로 넘기지 않고 **cross-room navigation**을 먼저 시도한다 (5장 참고). 안 가본 방이 없거나 갈 수 없으면 그제서야 `missionN_pipe.on_job_result(kind="exploration", route=[])`로 넘어가 미션별 최종 처리(카운트 확정 / FAILED)를 함.

---

## 3. Perception Pipeline (`perception/perception_pipeline.py::process()`)

`OBSERVE`/`FOLLOW_EXPLORATION`(이동 중 재관측) 상태에서 이미지+LiDAR가 준비되면 실행:

1. **YOLO-World 탐지** (`detector.py`) — `task["detection_prompts"]`(문장에서 뽑힌 카테고리들)로 open-vocab 탐지.
2. **낮은 confidence 재검증** (`detection_verifier.py`) — `config.DETECTION_VERIFICATION_CONFIDENCE_THRESHOLD`(0.35) 미만인 탐지만 모아 Gemini에게 한 번에 확인, fail-**open**(VLM 실패해도 원래 탐지 유지 — 놓치는 것보다 오탐 하나 받아들이는 게 나음).
3. **SAM2 세그멘테이션** (`segmenter.py`) — 2D mask 생성.
4. **LiDAR grounding** (`lidar_grounding.py::PanoramaLidarGrounder.ground()`) — SAM2 mask 안에 들어온 LiDAR point로 3D 위치 계산. **2단계 등급**:
   - `GROUNDING_MIN_POINTS`(5) 이상 → **precise**: percentile(5~95%) 기반 정밀 bbox.
   - `GROUNDING_MIN_POINTS_APPROXIMATE`(1) ~ 4개 → **approximate**: 중앙값 위치 + 기본 크기(`GROUNDING_APPROXIMATE_DEFAULT_SIZE_M`). 유리창처럼 LiDAR 반사가 잘 안 되는 물체 대응(카테고리 하드코딩 아니라 point 개수 기준 일반 규칙).
   - 0개 → 드롭 (방향/깊이 단서 자체가 없음).
5. **object_memory 갱신** (`memory/object_memory.py::ObjectMemory.update()`) — 같은 카테고리 기존 노드와 위치/외형 유사도로 매칭(`memory/object_association.py`), 매칭되면 지수이동평균으로 위치/크기 갱신(merge), 새 물체면 새 노드 생성. approximate 위치도 나중에 더 좋은 관측이 오면 이 평균으로 자연스럽게 정밀해짐.

---

## 4. Scene Graph & 물체 간 관계 (`scene_graph/scene_graph_manager.py`)

### 노드/엣지 구조 (SysNav 논문 Sec IV-A-1)
- **Object Node**: `object_memory`의 각 물체. `self_attributes`(color 등, on-demand VLM 추론) 캐시 포함.
- **Viewpoint Node**: LiDAR coverage가 기존 대비 `VIEWPOINT_NOVEL_VOXEL_THRESHOLD` 이상 새로울 때만 생성(논문의 novelty 기준). 대표 이미지 저장(`scene_graph_viewpoints/`).
- **Room Node**: 5장 참고.

### 속성(attribute) 검증 — `reasoning/attribute_verifier.py`
문장에 색상 등 속성 제약이 있으면, 후보가 1개뿐이어도 반드시 VLM으로 확인(예전엔 "후보 1개면 그냥 확정"하던 지름길이 색을 안 보고 넘어가는 버그였음). Fail-**closed**(확인 안 되면 불통과 — attribute 확인이 핵심인데 실패를 통과로 치면 안 됨).

### 물체 간 관계(relation) 검증 — **3단계 폴백** (핵심)
문장에 "closest to"/"near"/"between" 같은 관계 제약이 있을 때, 어떻게든 검증해서 `scene_graph`에 object-object edge를 만든다. `scene_graph_manager._sync_objects`/`find_matching_target_ids`가 이 edge들을 relation_chain 순서대로 따라가며 최종 후보를 찾음.

1. **같은 프레임 동시 관측 + Gemini** (`_infer_task_relations_from_common_viewpoints`, `reasoning/spatial_relation_reasoner.py::infer()`) — 기존 경로. 두 물체가 같은 viewpoint에서 함께 관측된 적 있어야 시도됨. 이미지를 Gemini에게 보여주고 관계가 맞는지 확인, 실패 시 순수 기하 계산(`_infer_with_geometry`)으로 폴백.
2. **전역 위치 기반 순수 기하** (`_infer_task_relations_globally`, `spatial_relation_reasoner.py::infer_global()`) — **같은 프레임일 필요 없음**. object_memory에 있는 전역 3D 위치만으로 관계 판정(Lang2LTL-2 논문의 Spatial Predicate Grounding 방식: figure/ground를 독립적으로 grounding한 뒤 벡터 기하로 relation 판정). `near`/`nearest`/`between`/`on`/`above`/`under`/`beside`는 로봇 pose도 불필요. 두 물체가 각각 언제든 한 번이라도(precise든 approximate든) grounding만 됐으면 성립.
3. **후보 자신의 이미지로 VLM 직접 확인** (`reasoning/relation_image_verifier.py::verify()`, `sysnav_node.py::selection_job`) — 참조 물체가 3D로 **아예** 안 잡혀도(0 point) 됨. 후보(예: bedside table)의 대표 이미지를 Gemini에게 보여주고 "이 사진에 [참조 물체]가 [관계]로 보이는가?"를 직접 확인. `attribute_verifier`와 같은 on-demand 패턴을 관계형 predicate로 확장한 것. Fail-closed.

1번이 안 되면 2번, 2번도 안 되면(참조 물체가 전역 위치조차 없으면) 3번까지 자동으로 넘어간다. 셋 다 실패하면 `relation_pending: True`로 확정을 미루고 계속 탐색.

### "nearest" 최상급 판정 버그(수정됨)
`_infer_nearest_with_geometry`가 예전엔 target object_id로 그룹핑해서, 참조 카테고리 인스턴스가 2개 이상(창문 2개)이면 각각 별도 그룹이 되어 둘 다 "승리"해버리는 버그가 있었음 → `(source_category, target_category)` 기준으로 그룹핑하도록 수정, 진짜 가장 가까운 인스턴스 하나만 선택되게 함.

---

## 5. Room 구조 & Cross-room Navigation (SysNav 논문 Sec IV-A-1, IV-B-2)

### Room 분할 + 분류
- `rooms/room_segmenter.py::RoomSegmenter.segment()` — occupancy grid에서 distance-transform+watershed로 방을 분할. 매 mapping cycle마다 **처음부터 다시 계산**(room_id가 사이클마다 바뀔 수 있음).
- `rooms/room_registry.py::RoomRegistry` — 위 결과를 centroid 매칭으로 **안정적인 persistent room_id**에 이어붙임. 각 방에 배정된 viewpoint 중 `coverage_voxel_count`가 가장 큰 것을 대표 이미지로 추적.
- `reasoning/room_classifier.py::RoomClassifier` — 방의 대표 이미지로 카테고리(kitchen/bedroom 등) on-demand VLM 추론, 캐싱. Fail-open 아님(실패하면 미분류 상태 유지, 다음 사이클 재시도).
- `rooms/room_visualizer.py` — `room_segmentation_latest.png`에 분할+카테고리 시각화.

### Cross-room navigation (`sysnav_node.py::_try_start_cross_room_navigation` / `_on_cross_room_select_result`)
`PLAN_EXPLORATION`이 빈 route를 반환하면(현재 알려진 영역에 더 볼 게 없음):
1. `RoomRegistry.unvisited_rooms()` — 기하학적으로는 분할됐지만 viewpoint가 하나도 없는(=로봇이 실제로 들어간 적 없는) 방 목록.
2. 이번 task에서 이미 시도해본 방(`self._cross_room_attempted_ids`)은 제외.
3. 있으면 `rooms/cross_room_navigator.py::select_job` 제출(worker thread) — 카테고리를 아는 방은 `reasoning/room_relevance_selector.py::RoomRelevanceSelector.rank()`(VLM, task 문장 기반 관련도 순위, fail-open→거리순)로 우선순위, 모르는 방은 거리순. 순서대로 `coverage_planner.plan_direct_path()`가 실제로 되는 첫 방을 선택.
4. 성공하면 그 방까지 가는 waypoint를 `exploration_route`에 넣고 기존 `FOLLOW_EXPLORATION`으로 이동 → 도착하면 그 방 기준으로 `PLAN_EXPLORATION` 재개.
5. 안 가본 방이 없거나 전부 실패하면 원래 "빈 route" 상황으로 되돌려 미션별 종료 처리.

---

## 6. 탐색 알고리즘 (`exploration/coverage_planner.py::plan_route()`, 논문 Sec IV-B-1)

1. Occupancy grid에서 로봇 clearance(`ROBOT_CLEARANCE_M`)만큼 벽을 부풀린 `traversable` 계산.
2. 논문의 surface point set **S**(free/non-free 경계, frontier) 추출.
3. 현재 **방** 안에서 무작위 pose 후보 샘플링(`EXPLORATION_CANDIDATE_SAMPLES`개) + 아직 안 풀린 frontier 컴포넌트마다 보장되는 **anchor** 후보(방 제한 없이 전체 맵에서 탐색 — 문 통과 문제 대응).
4. 각 후보의 **wcov**(반경 내 + line-of-sight로 보이는 아직-안-덮인 surface point 수) 계산.
5. Stochastic sampling(`EXPLORATION_STOCHASTIC_TRIALS`번 반복) + TSP로 비용 최소 경로 선택.
6. A* 경로를 `EXPLORATION_PATH_WAYPOINT_SPACING_M` 간격으로 잘라 중간 waypoint 생성.

### Anchor 무한루프 버그(수정됨)
Anchor는 `viewpoint_memory.is_near_visited()` 예외 대상(문 통과용)인데, 그 옆 unknown 셀이 **영영 안 풀리면**(예: 유리창이라 LiDAR가 못 뚫음) 매 사이클 같은 anchor가 다시 잡혀서 무한 루프에 빠졌음. `_anchor_visit_counts`로 같은 anchor가 몇 번 연속 잡히는지 세서, `EXPLORATION_ANCHOR_MAX_REVISITS`(5회) 넘으면 예외 자격을 박탈 → 결국 후보에서 빠져서 정상적으로 "더 볼 곳 없음"에 수렴하도록 수정.

---

## 7. Mission 1 — Numerical (`missions/mission1_pipe.py`)

- 파싱: Object Reference와 동일(`LLMQueryParser`).
- **후보를 찾아도 절대 멈추지 않음** — `_on_perception_result`가 `origin_state=="OBSERVE"`일 때만 `PLAN_EXPLORATION`으로 보내고, 후보 존재 여부는 무시.
- `plan_route()`가 빈 route를 반환(=탐색 완전히 소진, cross-room까지 다 시도한 뒤)하면 `MISSION1_FINALIZE_COUNT`로 전환(FAILED 아님 — "다 봤다"는 정상 종료 신호).
- `count_job` — `object_memory.find_by_category()` → relation 필터(`scene_graph.find_matching_target_ids`) → attribute 필터(`attribute_verifier`) 순서로 걸러서 개수 확정.
- `/numerical_response`(Int32) 발행, `SUCCESS`.

## 8. Mission 2 — Object Reference (`missions/mission2_pipe.py`)

- `SELECT_TARGET` → `selection_job`(위 4장의 attribute/relation 검증 전체 + `GeminiSelector.select()`로 최종 하나 확정) → 확정되면 `object_approach_pose()`로 접근 지점 계산, `goal_publisher.publish()` + **`/selected_object_marker`(Marker, 단수) 발행** → `NAVIGATE_TARGET`.
- `NAVIGATE_TARGET`에서 `goal_reached()`면 `SUCCESS`.
- README의 "marker 중심점이 곧 navigation waypoint" 요구사항대로 marker와 waypoint를 같은 시점(확정 시점)에 함께 발행.

## 9. Mission 3 — Instruction-Following (`missions/mission3_pipe.py`)

### 문장 → 절 분해 (`parse_instruction`)
1. `_split_clauses()` — 정규식 기반. 트리거: `go to/near/between`, `stop at/by`, `take the path`, `avoid(ing) the path`, `pass by`, 생략형 `and then to`/`and finally, to`. `questions.json` 30문장 전수 검증(생략된 동사, "go between A and B", "the two X" 집합형까지 포함).
2. 트리거를 하나도 못 찾으면 → **LLM 폴백**(`task/llm_instruction_splitter.py::LLMInstructionSplitter.split()`) — Gemini가 같은 구조로 절을 분해, 실패하면 빈 리스트(태스크 거부).
3. 각 절을 다음 중 하나로 분류:
   - **destination**(`is_stop=True`, 순서대로 실제 정지): 물체 카테고리 절이면 `LLMQueryParser.parse()`로 파싱(`resolve="category"`), "between A and B" 절이면 두 참조를 기하로 resolve(`resolve="point"`).
   - **positive_path**(`is_stop=False`): 지나가긴 해야 하지만 정지는 아님. destination과 동일하게 순서대로 waypoint 큐에 들어감.
   - **negative_path**: `task["global_forbidden"]`에 별도 저장 — 문장 어디에 있든 **남은 경로 전체에 적용되는 전역 제약**으로 취급(README의 "전체 궤적이 금지구역을 지나갔는지" 채점 기준에 맞춤).

### 실행 (`MISSION3_SELECT_STEP` ↔ `MISSION3_NAVIGATE_STEP`)
- `_select_step`: 현재 `steps[step_index]` 처리. `resolve="category"`면 `selection_job` 제출(4장 3단계 relation 검증 전체 재사용). `resolve="point"`면 `object_memory`에서 참조 카테고리 후보의 위치를 직접 가져와(VLM 없이, 가장 가까운 것) 기하 계산.
- 처리 중 매번 `_try_resolve_forbidden` 시도 — `global_forbidden`의 참조 물체 위치가 확보되면 A-B 선분(or 한 점) 주변에 `INSTRUCTION_FORBIDDEN_RADIUS_M` 반경으로 forbidden mask 생성(`node.mission3_forbidden_mask`).
- 목표 지점이 정해지면 `_start_navigate_to_point`: forbidden mask가 있으면 `coverage_planner.plan_direct_path()`(A*로 우회 경로) 사용, 없으면 단순 접근점 하나 직접 발행. `mission3_leg_queue`에 waypoint 채우고 `MISSION3_NAVIGATE_STEP`으로.
- 도착하면 `mission3_step_index += 1`, 다음 step 있으면 `MISSION3_SELECT_STEP`으로, 없으면 `SUCCESS`.
- `/way_point_with_heading`(Pose2D)을 순서대로 여러 번 발행 — 채점은 토픽 값이 아니라 **로봇이 실제로 그 경로를 따라간 궤적**을 봄(README).

---

## 10. 출력 토픽 요약

| 토픽 | 타입 | 미션 | 채점 기준 |
|---|---|---|---|
| `/numerical_response` | `std_msgs/Int32` | Numerical | 정확히 일치해야 1점(부분점수 없음) |
| `/selected_object_marker` | `visualization_msgs/Marker`(단수) | Object Reference | bbox 겹침 정도로 0~2점 |
| `/way_point_with_heading` | `geometry_msgs/Pose2D` 시퀀스 | Instruction-Following (+ 탐색 이동에도 공용 사용) | 실제 궤적(순서/제약달성/금지구역회피)으로 0~6점 |

`/sysnav/object_markers`(MarkerArray)는 채점 대상이 **아닌** 디버그 전용 토픽(전체 관측 물체 시각화).

---

## 11. 디버깅 도구

- **`ai_module/debug/mission_status_latest.html`** — 실시간 대시보드(`mission_dashboard.py`, 1초 갱신). 현재 미션/state, 10분 타임리밋 진행바, 미션별 상세(Numerical: 후보 수, Object Reference: 선택된 물체, Instruction-Following: 전체 plan + 진행 상태 + parser 종류), 마지막 발행값. `./docker/ui_checker.sh`로 열기.
- **`ai_module/debug/room_segmentation_latest.png`** — 방 분할 + 카테고리.
- **`ai_module/debug/scene_graph_latest.png/.json/.dot`** — Room/Viewpoint/Object 그래프.
- **`ai_module/debug/exploration_debug_latest.png`** — surface point(frontier) + 로봇 위치.
- **`ai_module/debug/sysnav_relation_check.txt`** — 관계 검증 시도 전부(통과/실패 포함) 기록.
- **`ai_module/debug/sysnav_detect_*.jpg`** — 매 perception마다 bbox+mask+3D 위치 오버레이.

---

## 12. 알려진 한계 / 다음 단계 후보

- `find_matching_target_ids`가 relation_chain을 hop 단위로 따라가는데, 다단계 체인(A closest to B near C)에서 각 hop이 서로 다른 시점/경로(co-observation/전역기하/이미지폴백)로 검증될 수 있어 조합 시 신뢰도가 균일하지 않음.
- `relation_image_verifier`는 "nearest" 같은 최상급을 후보별 독립 boolean으로만 판정(여러 후보가 동시에 "yes"일 수 있음) — 최종 선택은 `GeminiSelector`의 문장 전체 맥락 판단에 위임.
- Cross-room navigation은 문이 닫혀서 LiDAR가 아예 못 본 방은 애초에 `RoomRegistry`에 등록조차 안 되므로 여전히 못 찾음(물리적 한계).
- Mission 3의 `_split_clauses` 트리거 목록 밖의 표현은 LLM 폴백에 의존 — 폴백 자체가 검증 데이터셋(30문장) 밖이라 실전 정확도 미검증.
