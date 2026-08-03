# TF 좌표 변환 모듈

이 폴더는 RViz 확인용 코드가 아니라, 실제 파이프라인에서 좌표 변환을 담당한다.

현재 기준 TF 구조는 다음과 같다.

```text
map
 └─ sensor      position=(0,0,0.75), orientation=(0,0,0,1)
     └─ camera  position=(0,0,0.85), orientation=(-0.5,0.5,-0.5,0.5)
```

`coordinate_transform.py`는 먼저 ROS TF를 조회한다. TF 조회가 실패하면 `config.py`의
`STATIC_TF_FALLBACKS`를 사용한다.

파노라마 픽셀에서 만든 ray는 camera optical frame 기준이다.

```text
camera optical frame
 +x: image right
 +y: image down
 +z: camera forward
```

이 ray를 `CoordinateTransformer.transform_direction(ray, "camera", "map")`로
map frame 방향 벡터로 바꾼 뒤 PointCloud2와 매칭한다.
