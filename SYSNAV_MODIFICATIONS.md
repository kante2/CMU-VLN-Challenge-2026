# SysNav 수정 내역

이 문서는 `feature/parser` 브랜치의 현재 코드를 기준으로 지금까지 적용한 SysNav 변경사항을 정리한다.

## 1. Git 상태

- 작업 브랜치: `feature/parser`
- 원격 브랜치: `origin/feature/parser`
- 주요 커밋: `0765816 Add LLM alias parsing and robust object grounding`
- 커밋 이후 추가 수정사항은 아직 커밋되지 않은 working tree 변경으로 남아 있다.
- 사용자가 별도로 수정한 `docker/C_질의.sh`, `readme_kante2.md`, `scene/`은 SysNav 변경과 분리해서 관리해야 한다.

## 2. 질문 파싱 및 detection alias

### Gemini 질문 파싱

자연어 질문을 다음 구조로 변환하도록 Gemini parser를 추가했다.

```json
{
  "target": "bowl",
  "attributes": [],
  "relation": "nearest",
  "reference_objects": ["knife rack", "trash can"]
}
```

Gemini를 사용할 수 없거나 응답이 잘못되면 기존 rule parser를 fallback으로 사용한다.

### Detection alias 확장

canonical category 하나에 YOLO-World가 이해하기 쉬운 여러 prompt를 연결한다.

```text
knife rack
→ knife holder
→ knife block
→ kitchen knife holder
→ magnetic knife strip
→ wall-mounted knife rack
→ set of kitchen knives
```

YOLO-World가 alias로 검출한 결과는 다시 canonical category로 통합한다. Object Memory에는 canonical category와 실제 검출 prompt인 `detected_as`를 함께 저장한다.

### 이미지 기반 alias fallback

필수 category가 검출되지 않으면 Gemini에 현재 이미지를 전달해 추가 visual alias를 제안받고 YOLO-World를 한 번 더 실행한다.

## 3. LiDAR grounding

### 기본 grounding 기준

```text
GROUNDING_MIN_POINTS = 3
```

- 한 프레임에서 3점 이상이면 즉시 3D grounding한다.
- 한 프레임에서 1~2점이면 provisional 후보로 저장한다.
- 서로 다른 2개 이상의 프레임에서 누적 3점 이상이면 최종 등록한다.
- provisional 후보는 category와 3D 위치를 이용해 연결한다.
- 기본 만료 시간은 5초, association 거리는 0.75m이다.

### bbox와 SAM mask hit 분리

각 detection마다 다음 값을 별도로 계산한다.

- `bbox_hits`: detection bbox 안에 투영된 LiDAR point 수
- `mask_hits`: SAM mask 안에 투영된 LiDAR point 수
- `supplemented_hits`: bbox fallback으로 추가한 point 수
- `final_hits`: 실제 3D grounding에 사용한 최종 point 수

실시간 로그 예시는 다음과 같다.

```text
[sysnav lidar_grounding] category=bowl bbox_hits=3 mask_hits=0 supplemented_hits=3 final_hits=3
```

`flush=True`를 사용해 ROS launch 종료 시 한꺼번에 출력되던 buffering 문제를 수정했다.

### bbox 보조 fallback

현재 규칙은 다음과 같다.

```text
mask_hits >= 3
→ SAM mask point만 사용

mask_hits = 1~2, bbox_hits >= 3
→ mask에서 5px 이내이고 기존 mask point와 depth 차이가 0.3m 이하인 bbox point 보충

mask_hits = 0, bbox_hits >= 3
→ bbox 내부 LiDAR point 전체 사용

bbox_hits < 3
→ 기존 provisional 규칙 적용 또는 grounding 실패
```

grounding 결과에는 source를 다음 중 하나로 기록한다.

```text
single_frame
mask_bbox_fallback
bbox_only_fallback
multi_frame_provisional
```

> 주의: `bbox_only_fallback`은 SAM과 겹치지 않는 배경, 조리대 또는 벽의 point를 객체 point로 잘못 사용할 수 있다. 작은 물체를 더 잘 등록하는 대신 잘못된 3D 위치가 생성될 위험이 있다.

## 4. LiDAR debug 이미지

다음 projection 이미지를 `ai_module/debug`에 저장한다.

```text
sysnav_lidar_projection_<timestamp>.jpg
```

- cyan: 파노라마 이미지에 투영된 전체 LiDAR point
- red: 해당 detection grounding에 선택된 point
- green: detection bbox와 category label

일반 detection 이미지에도 다음 정보를 표시한다.

```text
bowl 0.53 bbox=3 mask=0 +bbox=3
```

## 5. 객체 관계 판단

현재 관계 판단은 VLM과 3D geometry를 함께 사용하는 구조다.

```text
질문의 relation chain 파싱
→ 같은 viewpoint에 함께 저장된 객체 검색
→ Gemini VLM으로 관계 검증
→ VLM 결과가 없거나 API 호출에 실패하면 3D geometry fallback
→ 검증된 relation을 Scene Graph edge로 저장
```

geometry fallback은 다음 관계를 지원한다.

- `nearest`, `closest`: 후보와 reference 사이 XY 거리의 argmin
- `near`, `beside`: XY 거리와 객체 크기
- `left_of`, `right_of`, `in_front_of`, `behind`: viewpoint-local 좌표
- `above`, `under`, `on`: 3D bbox와 높이
- `between`: 두 reference를 연결한 선분과 target 사이 거리

관계 판단 결과는 다음 파일에서 확인할 수 있다.

```bash
tail -f /home/docker/ai_module/debug/sysnav_relation_check.txt
```

`method=gemini` 또는 `method=geometry`로 실제 사용된 판단 방식을 확인할 수 있다.

## 6. Target navigation

### 이동 상태 로그

`NAVIGATE_TARGET` 상태에서 1초마다 현재 위치, 목표, 거리와 목표 재발행 횟수를 출력한다.

```text
NAVIGATE_TARGET: robot=(-5.39, -1.66), goal=(-6.36, -2.16), dist=1.10m, republish_count=2
```

### 정체 감지 및 목표 재발행

- 8초 동안 목표 거리가 최소 0.1m 줄지 않으면 정체로 판단한다.
- 정체 시 동일 target goal을 다시 발행한다.
- 재발행 횟수를 로그에 남긴다.

```text
NAVIGATE_TARGET made no progress for 8s; republishing goal (...), attempt=3
```

### Target 성공 거리

탐색 waypoint와 target 성공 기준을 분리했다.

```text
Exploration waypoint: 0.55m
Target navigation:     1.50m
```

로봇이 target goal에서 1.5m 이내에 들어오면 성공 처리한다.

### Task 종료 로그

성공 시 다음 로그를 출력한다.

```text
TASK END 🏁 Target navigation completed (task_id=1)
```

## 7. 주요 설정값

| 설정 | 현재 값 | 의미 |
|---|---:|---|
| `GROUNDING_MIN_POINTS` | 3 | 즉시 grounding에 필요한 point 수 |
| `GROUNDING_PROVISIONAL_MIN_POINTS` | 1 | provisional 관측 최소 point 수 |
| `GROUNDING_PROVISIONAL_MIN_FRAMES` | 2 | provisional 승격 최소 프레임 수 |
| `GROUNDING_BBOX_FALLBACK_MAX_MASK_DISTANCE_PX` | 5px | sparse mask에서 허용하는 bbox 보조 거리 |
| `GROUNDING_BBOX_FALLBACK_DEPTH_TOLERANCE_M` | 0.30m | bbox 보조 point depth 허용 오차 |
| `GOAL_REACHED_DISTANCE_M` | 0.55m | exploration waypoint 도착 기준 |
| `TARGET_GOAL_REACHED_DISTANCE_M` | 1.50m | target 도착 성공 기준 |
| `TARGET_STATUS_LOG_INTERVAL_SEC` | 1초 | target 이동 상태 로그 주기 |
| `TARGET_STUCK_TIMEOUT_SEC` | 8초 | target 이동 정체 판단 시간 |
| `TARGET_STUCK_PROGRESS_M` | 0.10m | 진행으로 인정할 최소 거리 감소량 |

## 8. 테스트 및 빌드

LiDAR grounding 테스트에는 다음 항목이 포함된다.

- bbox 내부 projected point 계산
- sparse mask에 depth-consistent bbox point 보충
- `mask_hits=0`일 때 bbox-only fallback
- bbox point가 3개 미만일 때 fallback 거부
- 서로 다른 프레임의 provisional 누적
- 같은 프레임의 중복 detection 누적 방지

컨테이너 빌드 명령:

```bash
docker exec iros2026_sysnav_module bash -lc '
source /opt/ros/jazzy/setup.bash &&
cd /home/docker/ai_module &&
colcon build --symlink-install --packages-select sysnav
'
```

전체 테스트 명령:

```bash
docker exec iros2026_sysnav_module bash -lc '
source /opt/ros/jazzy/setup.bash &&
cd /home/docker/ai_module/src/sysnav_ros2_mvp &&
PYTHONPATH=.:$PYTHONPATH /usr/bin/python3 -m unittest discover -s tests -v
'
```

## 9. 실행 방법

빌드 후 실행 중인 SysNav를 종료하고 다음 명령으로 재시작한다.

```bash
source /opt/ros/jazzy/setup.bash
source /home/docker/ai_module/install/setup.bash
ros2 launch sysnav sysnav.launch.py
```

질문 발행 예시:

```bash
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'find the bowl closest to the knife rack near the trash can.'}"
```

## 10. 알려진 위험 및 후속 개선점

1. `bbox_only_fallback`은 객체가 아닌 배경 point를 사용할 수 있다.
2. 잘못된 bbox-only 3D 위치는 geometry relation 판단과 target goal을 왜곡할 수 있다.
3. target goal 재발행은 유실된 goal에는 효과가 있지만, 목표가 장애물 안에 있으면 반복 재발행만 발생한다.
4. target 접근 goal을 occupancy map의 free cell로 보정하는 로직이 추가로 필요하다.
5. VLM 관계 API 실패와 빈 응답을 더 자세히 로깅하면 geometry fallback 원인을 구분하기 쉽다.
