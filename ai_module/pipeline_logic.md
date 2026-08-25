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
│           └─ 빈 route면 미션별 종료 처리로                    │
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
(`ai_module/docker/ui_checker.sh`로 열기, `mission_dashboard.py`가 매 사이클 갱신).

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
지도 원점/로봇 cell이 아직 준비 안 된 일시적 실패(`origin_not_ready` 등)면 `PLAN_EXPLORATION`으로 되돌린다. 그 외에는 탐색이 완전히 소진된 것으로 보고 `missionN_pipe.on_job_result(kind="exploration", route=[])`로 넘어가 미션별 최종 처리(카운트 확정 / FAILED)를 함.

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
- **Object Node**: `object_memory`의 각 물체. VLM 판정 캐시 두 종류를 들고 있다 — `self_attributes`(color 등, `attribute_verifier`)와 `relation_checks`(이미지 기반 관계 판정, `relation_image_verifier`). 2D bbox 크롭도 두 장 보관: `representative_image`(배경 제거, 속성 판정용)와 `context_image`(배경 유지 + 여유, 관계 판정용). 사진이 교체되면 `image_version`이 올라가고, 그게 `relation_checks` 캐시의 무효화 신호다.
- **Viewpoint Node**: LiDAR coverage가 기존 대비 `VIEWPOINT_NOVEL_VOXEL_THRESHOLD` 이상 새로울 때만 생성(논문의 novelty 기준). 대표 이미지 저장(`scene_graph_viewpoints/`).
- **Room Node**: 호환성용 고정 노드 `Room_0` 하나(`SCENE_GRAPH_SINGLE_ROOM_ID`). 모든 Viewpoint/Object가 여기에 매달린다.

### 속성(attribute) 검증 — `reasoning/attribute_verifier.py`
문장에 색상 등 속성 제약이 있으면, 후보가 1개뿐이어도 반드시 VLM으로 확인(예전엔 "후보 1개면 그냥 확정"하던 지름길이 색을 안 보고 넘어가는 버그였음). Fail-**closed**(확인 안 되면 불통과 — attribute 확인이 핵심인데 실패를 통과로 치면 안 됨).

### 물체 간 관계(relation) 검증 — **3단계 폴백** (핵심)
문장에 "closest to"/"near"/"between" 같은 관계 제약이 있을 때, 어떻게든 검증해서 `scene_graph`에 object-object edge를 만든다. `scene_graph_manager._sync_objects`/`find_matching_target_ids`가 이 edge들을 relation_chain 순서대로 따라가며 최종 후보를 찾음.

1. **같은 프레임 동시 관측 + Gemini** (`_infer_task_relations_from_common_viewpoints`, `reasoning/spatial_relation_reasoner.py::infer()`) — 기존 경로. 두 물체가 같은 viewpoint에서 함께 관측된 적 있어야 시도됨. 이미지를 Gemini에게 보여주고 관계가 맞는지 확인, 실패 시 순수 기하 계산(`_infer_with_geometry`)으로 폴백.
2. **전역 위치 기반 순수 기하** (`_infer_task_relations_globally`, `spatial_relation_reasoner.py::infer_global()`) — **같은 프레임일 필요 없음**. object_memory에 있는 전역 3D 위치만으로 관계 판정(Lang2LTL-2 논문의 Spatial Predicate Grounding 방식: figure/ground를 독립적으로 grounding한 뒤 벡터 기하로 relation 판정). `near`/`nearest`/`between`/`on`/`above`/`under`/`beside`는 로봇 pose도 불필요. 두 물체가 각각 언제든 한 번이라도(precise든 approximate든) grounding만 됐으면 성립.
3. **후보 자신의 이미지로 VLM 직접 확인** (`reasoning/relation_image_verifier.py::verify()`, `sysnav_node.py::selection_job`) — 참조 물체가 3D로 **아예** 안 잡혀도(0 point) 됨. 후보(예: bedside table)의 `context_image`를 Gemini에게 보여주고 "이 사진에 [참조 물체]가 [관계]로 보이는가?"를 직접 확인. 최상급(`nearest`/`farthest`)은 후보마다 독립 yes/no를 물으면 둘 다 통과할 수 있어서, `rank_superlative()`가 후보 전부를 한 번에 놓고 비교시킨다. `attribute_verifier`와 같은 on-demand 패턴을 관계형 predicate로 확장한 것. Fail-closed.
   - **캐싱(2026-08-24 추가)**: 판정 결과는 object node의 `relation_checks`에 적립되고 키에 `image_version`이 들어간다 — **사진이 교체될 때만** 다시 묻는다. 이게 없을 때는, 참조 물체가 끝내 grounding 안 되는 경우(이 폴백이 존재하는 바로 그 상황) `relation_pending` → `PLAN_EXPLORATION` → `OBSERVE` → `SELECT_TARGET` 사이클마다 같은 사진을 같은 질문으로 계속 올렸다(mission3는 step마다 새로 시작해서 더 심함). 최상급은 집합 전체를 놓고 내린 판정이라 키에 참가자 전원의 `(id, image_version)`이 들어가고, 참가한 모든 노드에 같은 키로 적립된다.

1번이 안 되면 2번, 2번도 안 되면(참조 물체가 전역 위치조차 없으면) 3번까지 자동으로 넘어간다. 셋 다 실패하면 `relation_pending: True`로 확정을 미루고 계속 탐색.

### "nearest" 최상급 판정 버그(수정됨)
`_infer_nearest_with_geometry`가 예전엔 target object_id로 그룹핑해서, 참조 카테고리 인스턴스가 2개 이상(창문 2개)이면 각각 별도 그룹이 되어 둘 다 "승리"해버리는 버그가 있었음 → `(source_category, target_category)` 기준으로 그룹핑하도록 수정, 진짜 가장 가까운 인스턴스 하나만 선택되게 함.

---

## 5. 탐색 알고리즘 (`exploration/coverage_planner.py::plan_route()`, 논문 Sec IV-B-1)

1. Occupancy grid에서 로봇 clearance(`ROBOT_CLEARANCE_M`)만큼 벽을 부풀린 `traversable` 계산.
2. 논문의 surface point set **S**(free/non-free 경계, frontier) 추출.
3. 알려진 맵 전체의 traversable cell에서 무작위 pose 후보 샘플링(`EXPLORATION_CANDIDATE_SAMPLES`개) + 아직 안 풀린 frontier 컴포넌트마다 보장되는 **anchor** 후보(작고 먼 frontier를 샘플링 운으로 놓치지 않게).
4. 각 후보의 **wcov**(반경 내 + line-of-sight로 보이는 아직-안-덮인 surface point 수) 계산.
5. Stochastic sampling(`EXPLORATION_STOCHASTIC_TRIALS`번 반복) + TSP로 비용 최소 경로 선택.
6. A* 경로를 `EXPLORATION_PATH_WAYPOINT_SPACING_M` 간격으로 잘라 중간 waypoint 생성.

### 발행 거부 라이브락(수정됨, 2026-08-24)
`plan_route()`가 route를 반환해도 `goal_publisher.publish()`가 그 hop을 **하나도 발행 못 하는** 경우가 있다(base autonomy의 `waypointConverter`가 받아줄 travArea 점이 그 좌표 근처에 없음 → `SNAP_FAIL` / `SNAP_NO_PROGRESS`). 그러면 `publish_next_exploration_goal()`이 `OBSERVE`로 되돌리는데, 로봇이 안 움직이니 지도가 그대로고 → 다음 사이클에 **완전히 동일한 route**가 다시 나온다. `plan_route()`는 빈 route를 반환하지 않으므로 미션별 종료 처리가 영원히 안 불린다(실측: 1.2초 주기 무한루프. Mission 2는 `MISSION2_EXPLORATION_TIME_LIMIT_SEC`이 결국 구해주지만 Mission 1/3은 탈출구가 없음).

방어선 두 개:
1. **planner blacklist** — 발행 거부된 좌표마다 `CoveragePlanner.mark_unpublishable()`로 셀에 strike를 적립하고, `EXPLORATION_UNPUBLISHABLE_MAX_STRIKES`(2)를 넘긴 셀은 후보 풀에서 뺀다(anchor도 예외 없음). 1이 아니라 2인 이유는 terrain_map이 롤링 윈도우라 지금 멀어서 못 받는 지점도 가까이 가면 받아줄 수 있기 때문. `reset()`(새 질문)에서 초기화.
2. **streak 승격** — "route는 나왔는데 전 hop 발행 거부"가 `EXPLORATION_UNPUBLISHABLE_ROUTE_LIMIT`(5회) 연속되면 `_finish_exploration_as_exhausted()`가 빈 route와 동일하게 미션별 종료 처리로 넘긴다. 넓은 공간에서는 매 사이클 후보를 새로 무작위 샘플링해서 1번만으로는 안 마르기 때문에 필요하다.

### Anchor 무한루프 버그(수정됨)
Anchor는 `viewpoint_memory.is_near_visited()` 예외 대상(문 통과용)인데, 그 옆 unknown 셀이 **영영 안 풀리면**(예: 유리창이라 LiDAR가 못 뚫음) 매 사이클 같은 anchor가 다시 잡혀서 무한 루프에 빠졌음. `_anchor_visit_counts`로 같은 anchor가 몇 번 연속 잡히는지 세서, `EXPLORATION_ANCHOR_MAX_REVISITS`(5회) 넘으면 예외 자격을 박탈 → 결국 후보에서 빠져서 정상적으로 "더 볼 곳 없음"에 수렴하도록 수정.

---

## 6. Mission 1 — Numerical (`missions/mission1_pipe.py`)

- 파싱: Object Reference와 동일(`LLMQueryParser`).
- **후보를 찾아도 절대 멈추지 않음** — `_on_perception_result`가 `origin_state=="OBSERVE"`일 때만 `PLAN_EXPLORATION`으로 보내고, 후보 존재 여부는 무시.
- `plan_route()`가 빈 route를 반환(=탐색 완전히 소진)하면 `MISSION1_FINALIZE_COUNT`로 전환(FAILED 아님 — "다 봤다"는 정상 종료 신호).
- `count_job` — 개수를 **두 겹**으로 낸다.
  1. **기하 기반**: `object_memory.find_by_category()` → relation 필터(`scene_graph.find_matching_target_ids`, 여기선 edge 조회만 하고 추론은 안 돌린다) → attribute 필터(`attribute_verifier`) → `len()`.
  2. **VLM 기반**(`reasoning/vlm_counter.py`, `NUMERICAL_VLM_COUNT_ENABLED` 기본 on): 대상 카테고리 물체를 **가장 많이 동시에 본 viewpoint 한 장**(`scene_graph.best_viewpoint_for_objects()`)의 파노라마를 Gemini에게 보여 직접 세게 하고, 성공하면 그 값을 쓴다.
  - 왜 2가 필요한가: 1은 **탐지 재현율이 그대로 상한**이다 — 실측(home_building_1)에서 pillow가 GT 18개인데 메모리엔 7개만 남았다. 못 본 물체는 병합·필터를 아무리 손봐도 셀 수 없다.
  - 뷰를 **한 장으로 확정**하는 이유: 여러 뷰의 개수를 합치면 같은 물체가 여러 뷰에 찍혀 중복 계산된다. 뷰 선정은 relation/attribute 필터 **전의** 카테고리 전체로 하고(걸러진 물체도 이미지엔 찍혀 있다), 제약 판정은 질문 원문을 그대로 넘겨 VLM이 이미지에서 직접 본다.
  - 같은 이미지를 `NUMERICAL_VLM_COUNT_SAMPLES`(5)회 **병렬** 질의해 **최빈 개수**를 채택한다(self-consistency) — `temperature=0.0`인데도 응답이 결정적이지 않아 1회 호출은 사실상 동전던지기다. 동률이면 작은 쪽(파노라마 좌우 wrap 중복 계수가 흔한 오류).
  - 숫자 대신 **항목 목록**(`{"where": ...}`)을 받는다 — VLM은 5~6개를 넘으면 총합을 자주 틀리지만 나열은 비교적 안정적이고, 무엇을 셌는지 로그로 사람이 검증할 수 있다.
  - **fail-quiet**: 키 없음/에러/이미지 없음이면 조용히 1의 값을 쓴다. 개수 미션은 0/1 채점이라 "답을 못 냄"이 최악이다. 모델은 `GEMINI_COUNTING_MODEL`(기본 flash) → 실패 시 `GEMINI_MODEL` 순으로 시도하고, 호출당 `GEMINI_COUNTING_TIMEOUT_SEC`(45초) 상한을 둔다.
- `/numerical_response`(Int32) 발행, `SUCCESS`.

## 7. Mission 2 — Object Reference (`missions/mission2_pipe.py`)

- `SELECT_TARGET` → `selection_job`(위 4장의 attribute/relation 검증 전체 + `GeminiSelector.select()`로 최종 하나 확정) → 확정되면 `object_approach_pose()`로 접근 지점 계산, `goal_publisher.publish()` + **`/selected_object_marker`(Marker, 단수) 발행** → `NAVIGATE_TARGET`.
- `NAVIGATE_TARGET`에서 `goal_reached()`면 `SUCCESS`.
- README의 "marker 중심점이 곧 navigation waypoint" 요구사항대로 marker와 waypoint를 같은 시점(확정 시점)에 함께 발행.

## 8. Mission 3 — Instruction-Following (`missions/mission3_pipe.py`)

### 문장 → 절 분해 (`parse_instruction`)
1. `_split_clauses()` — 정규식 기반. 트리거: `go to/near/between`, `stop at/by`, `take the path`, `avoid(ing) the path`, `pass by`, 생략형 `and then to`/`and finally, to`. `questions.json` 30문장 전수 검증(생략된 동사, "go between A and B", "the two X" 집합형까지 포함).
2. 트리거를 하나도 못 찾으면 → **LLM 폴백**(`task/llm_instruction_splitter.py::LLMInstructionSplitter.split()`) — Gemini가 같은 구조로 절을 분해, 실패하면 빈 리스트(태스크 거부).
3. 각 절을 다음 중 하나로 분류:
   - **destination**(`is_stop=True`, 순서대로 실제 정지): 물체 카테고리 절이면 `LLMQueryParser.parse()`로 파싱(`resolve="category"`), "between A and B" 절이면 두 참조를 기하로 resolve(`resolve="point"`).
   - **positive_path**(`is_stop=False`): 지나가긴 해야 하지만 정지는 아님. destination과 동일하게 순서대로 waypoint 큐에 들어감.
   - **negative_path**: `task["global_forbidden"]`에 별도 저장 — 문장 어디에 있든 **남은 경로 전체에 적용되는 전역 제약**으로 취급(README의 "전체 궤적이 금지구역을 지나갔는지" 채점 기준에 맞춤).

### 실행 (`MISSION3_SELECT_STEP` ↔ `MISSION3_NAVIGATE_STEP`)
- `_select_step`: 현재 `steps[step_index]` 처리. `resolve="category"`면 `selection_job` 제출(4장 3단계 relation 검증 전체 재사용). `resolve="point"`면 `object_memory`에서 참조 카테고리 후보의 위치를 직접 가져와(VLM 없이, 가장 가까운 것) 기하 계산. 단, 이 step이 필요로 하는 카테고리(target + 관계 참조)를 아직 못 봤으면(`_missing_categories`) 판정을 시도하지 않고 먼저 탐사한다 — 안 그러면 selection_job의 이미지 폴백이 "아직 안 가봐서 못 본" 것까지 성급히 확정해버린다.
- **pending 시 즉시 확정 (`_resolve_pending_step`, 2026-08-24)**: `selection_job`이 `relation_pending`/`attribute_pending`/`verification_pending`을 돌려줬을 때, **이 step에 필요한 카테고리가 전부 관측돼 있으면 탐사로 돌아가지 않고 기하로 목적지를 정해 바로 goal을 찍는다**(`_best_effort_step_target`: `near`/`beside` 등은 참조 물체와 XY 거리가 최소인 후보, `farthest`는 최대인 후보; 참조가 3D 위치를 못 가졌으면 로봇에서 가장 가까운 후보 + "관계 미적용" 경고 로그). 아직 못 본 물체가 있을 때만 예전처럼 `PLAN_EXPLORATION`으로 간다.
  - 왜: `selection_job`의 유일한 탈출구인 `mission2_exploration_deadline_reached`는 **Mission 2 전용**이라 Mission 3에서는 절대 안 선다. 그래서 관계가 끝내 검증 안 되면(참조가 유리창이라 3D grounding 실패, Gemini `final_verification`이 계속 거절 등) 타겟도 참조도 눈앞에 있는데 탐사가 소진될 때까지 돌다가 FAILED로 끝났다. Mission 3 채점은 subgoal 순서 + 실제 궤적 + 부분점수라, "정확할 때까지 안 움직인다"보다 "지금 아는 것으로 가장 그럴듯한 곳에 바로 간다"가 항상 낫다.
- 처리 중 매번 `_try_resolve_forbidden` 시도 — `global_forbidden`의 참조 물체 위치가 확보되면 A-B 선분(or 한 점) 주변에 `INSTRUCTION_FORBIDDEN_RADIUS_M` 반경으로 forbidden mask 생성(`node.mission3_forbidden_mask`).
- **접근점 선정 (2026-08-24 개정)**: `terrain_monitor.choose_approach_point()`가 물체 주변을 링(반경 × 7각도)으로 샘플링해 base autonomy가 받아줄 지점을 찾는다. Mission 3는 `MISSION3_OBJECT_APPROACH_MAX_M`(0.9m) 상한이 있어 링이 하나뿐 = **후보 7개**다. 이 상한은 탐색 범위가 아니라 "물체 0.9m 안에 서야 `go to`를 수행한 것"이라는 **의미 규칙**이라 평상시엔 반드시 지킨다.
  - 링이 전부 실패하면 예전엔 terrain을 아예 안 보는 고정 standoff로 폴백해 **명령 불가한 좌표**를 잡았고, mission3가 그걸 무한 재발행했다(실측 2026-08-24: 변기 앞 0.5초 주기 정지. `probe_waypoint_push.py` 결과 — 이 씬은 travArea의 **7.1%**만 clearance ≥ 0.75m를 통과, 목표 좌표 자체는 0.42m, 스냅하면 로봇 0.36m 앞으로 떨어져 갈 거리가 없음).
  - 이제 `_navigate_step`이 `MISSION3_SUBGOAL_MAX_RETRIES`(20회 ≈ 10초) 연속 발행 실패 시 **포기 직전 딱 한 번** `allow_relaxed=True`로 재시도한다. 이 모드는 링 대신 **commandable set을 직접 훑어** 물체에 가장 가까운 통과 지점을 결정론적으로 고른다(상한 `TERRAIN_APPROACH_FALLBACK_MAX_M`=3.0m, 벽 너머 오검출 방지). 완화되는 것은 **물체까지의 거리뿐**이고 0.75m 클리어런스는 base autonomy의 하드 필터라 못 푼다.
  - 그마저 실패하면 `_give_up_step()`이 그 step을 포기하고 다음으로 넘어간다 — 한 step에 갇혀 남은 step을 통째로 잃는 것보다 부분점수가 낫다. 로그에 `GIVEN UP`으로 명시해 "지나갔다"와 구분한다.
- 목표 지점이 정해지면 `_start_navigate_to_point`: forbidden mask가 있으면 `coverage_planner.plan_direct_path()`(A*로 우회 경로) 사용, 없으면 단순 접근점 하나 직접 발행. `mission3_leg_queue`에 waypoint 채우고 `MISSION3_NAVIGATE_STEP`으로.
- 도착하면 `mission3_step_index += 1`, 다음 step 있으면 `MISSION3_SELECT_STEP`으로, 없으면 `SUCCESS`.
- `/way_point_with_heading`(Pose2D)을 순서대로 여러 번 발행 — 채점은 토픽 값이 아니라 **로봇이 실제로 그 경로를 따라간 궤적**을 봄(README).

---

## 9. 출력 토픽 요약

| 토픽 | 타입 | 미션 | 채점 기준 |
|---|---|---|---|
| `/numerical_response` | `std_msgs/Int32` | Numerical | 정확히 일치해야 1점(부분점수 없음) |
| `/selected_object_marker` | `visualization_msgs/Marker`(단수) | Object Reference | bbox 겹침 정도로 0~2점 |
| `/way_point_with_heading` | `geometry_msgs/Pose2D` 시퀀스 | Instruction-Following (+ 탐색 이동에도 공용 사용) | 실제 궤적(순서/제약달성/금지구역회피)으로 0~6점 |

`/sysnav/object_markers`(MarkerArray)는 채점 대상이 **아닌** 디버그 전용 토픽(전체 관측 물체 시각화).

---

## 10. 디버깅 도구

- **`ai_module/debug/mission_status_latest.html`** — 실시간 대시보드(`mission_dashboard.py`, 1초 갱신). 현재 미션/state, 10분 타임리밋 진행바, 미션별 상세(Numerical: 후보 수, Object Reference: 선택된 물체, Instruction-Following: 전체 plan + 진행 상태 + parser 종류), 마지막 발행값. `./ai_module/docker/ui_checker.sh`로 열기.
- **`ai_module/debug/scene_graph_latest.png/.json/.dot`** — Viewpoint/Object 그래프.
- **`ai_module/debug/exploration_debug_latest.png`** — surface point(frontier) + 로봇 위치.
- **`ai_module/debug/sysnav_relation_check.txt`** — 관계 검증 시도 전부(통과/실패 포함) 기록.
- **`ai_module/debug/sysnav_detect_*.jpg`** — 매 perception마다 bbox+mask+3D 위치 오버레이.

---

## 11. 알려진 한계 / 다음 단계 후보

- `find_matching_target_ids`가 relation_chain을 hop 단위로 따라가는데, 다단계 체인(A closest to B near C)에서 각 hop이 서로 다른 시점/경로(co-observation/전역기하/이미지폴백)로 검증될 수 있어 조합 시 신뢰도가 균일하지 않음.
- `relation_image_verifier`는 "nearest" 같은 최상급을 후보별 독립 boolean으로만 판정(여러 후보가 동시에 "yes"일 수 있음) — 최종 선택은 `GeminiSelector`의 문장 전체 맥락 판단에 위임.
- Mission 3의 `_split_clauses` 트리거 목록 밖의 표현은 LLM 폴백에 의존 — 폴백 자체가 검증 데이터셋(30문장) 밖이라 실전 정확도 미검증.
