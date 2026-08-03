# SysNav Viewpoint 구현 메모

## 적용한 논문 로직

Scene Graph의 Viewpoint는 perception 프레임마다 만들지 않는다.

```text
C_prev = 기존 Viewpoint들의 coverage region 합집합
C_t = 현재 로봇 pose에서 관측된 coverage region
C_novel = C_t - C_prev

|C_novel| > omega 인 경우에만 새 Viewpoint 생성
```

새 Viewpoint 속성은 다음과 같다.

```text
p_i : map-frame robot pose
C_i : map-frame coverage voxel key 집합
I_i : 해당 pose의 panorama 이미지
```

## 현재 센서 코드에서 C_t를 만드는 방법

논문은 coverage distance와 novel coverage 조건을 설명하지만 voxel 크기, novel voxel
threshold 등의 구체적인 수치는 공개하지 않는다. 이 패키지는 360° LiDAR ray를 다음과
같이 voxelize한다.

```text
PointCloud2
  → T_SENSOR_TO_BASE 적용
  → robot pose로 map frame 변환
  → d_cover 안의 ray만 선택
  → sensor origin부터 endpoint까지 voxelize
  → C_t 생성
```

기본값은 `sysnav/config.py`와 환경변수에서 조절한다.

```text
SYSNAV_VIEWPOINT_COVERAGE_DISTANCE_M=4.0
SYSNAV_VIEWPOINT_COVERAGE_VOXEL_SIZE_M=0.40
SYSNAV_VIEWPOINT_NOVEL_VOXEL_THRESHOLD=120
SYSNAV_VIEWPOINT_COVERAGE_MAX_RAYS=1600
```

실제 LiDAR 밀도와 환경 크기에 따라 `NOVEL_VOXEL_THRESHOLD`를 우선 튜닝해야 한다.

## Object-Object Edge

공간 관계 질의가 들어오면 현재 프레임만 검사하지 않는다.

```text
Scene Graph Viewpoint 검색
  → target과 reference object를 모두 observes하는 Viewpoint 선택
  → 저장된 panorama image 로드
  → Gemini 관계 검증
  → 실패 시 해당 Viewpoint pose + 3D bbox geometry fallback
  → 관계가 성립하면 Object-Object Edge 생성
```

## Debug 출력

```text
DEBUG_DIR/
├── scene_graph_latest.json
├── scene_graph_latest.dot
├── scene_graph_latest.png
└── scene_graph_viewpoints/
```

`scene_graph_latest.json`의 각 Viewpoint에는 `coverage_region`,
`coverage_voxel_count`, `novel_voxel_count`가 저장된다.

## 현재 범위

- Viewpoint 생성 조건과 Viewpoint 속성은 coverage 기반으로 변경했다.
- 논문의 Multi-Room segmentation과 Room category 추론은 아직 포함하지 않는다.
- coverage 평가는 synchronized perception job이 실행될 때 수행한다. 모든 LiDAR scan마다
  별도 Viewpoint를 만드는 구조는 아니며, `PERCEPTION_WHILE_MOVING_INTERVAL_SEC`의 영향을
  받는다.
