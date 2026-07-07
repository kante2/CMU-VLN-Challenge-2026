# 1. 컨테이너 안으로 들어가서
docker exec -it iros2026_system bash

# 2. 컨테이너 안에서 (프롬프트가 docker@ 로 바뀐 뒤) 실행
pkill -f Model.x86_64
pkill -f vehicle_simulator
pkill -f rviz2
pkill -f ros2