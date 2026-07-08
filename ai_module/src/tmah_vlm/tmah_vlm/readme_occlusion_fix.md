# 3D target occlusion fix

기존 방식은 selected 2D box의 중심 ray와 가까운 LiDAR point를 고르는 방식이었다.
이 방식은 TV처럼 뒤에 있는 물체를 선택했을 때, ray 근처의 앞 선반/테이블 point가 더 가까워서 잘못 선택될 수 있다.

수정 방식:

1. PointCloud2를 map frame으로 변환한다.
2. 다시 camera frame으로 변환한다.
3. camera point를 panorama pixel로 투영한다.
4. selected 2D detection box 안에 들어온 point만 3D 후보로 사용한다.
5. 후보 point를 depth cluster로 나눈다.
6. 가까운 점을 무조건 선택하지 않고, box 중심과 잘 맞는 cluster의 대표점을 target으로 사용한다.

핵심 파일:

- `grounding/projector.py`
- `config.py`의 `BBOX_*` 설정값
