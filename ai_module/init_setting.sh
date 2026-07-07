# 1. 컨테이너 안으로 들어가서
docker exec -it iros2026_system bash

# 2. 컨테이너 안에서 (프롬프트가 docker@ 로 바뀐 뒤) 실행
pkill -9 -f autonomy_stack_mecanum_wheel_platform
pkill -9 -f static_transform_publisher
pkill -9 -f joy_node
pkill -9 -f default_server_endpoint