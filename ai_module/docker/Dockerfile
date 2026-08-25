# CMU VLN Challenge 2026 - 팀 제출용 AI 모듈 이미지
#
# 루트 docker/compose.yml(대회 원본, 수정 금지)의 ai_module 서비스가 빌드하는 파일이
# 바로 이 Dockerfile이다. 따라서 우리 모델은 여기서 빌드돼야 평가 환경에서 실행된다.
#
# 무거운 의존성(torch / SAM2 / ultralytics / CLIP / 모델 가중치)은 Docker Hub에 올려둔
# 이미지에 이미 들어있으므로 여기서 다시 설치하지 않는다. 레포의 sysnav 소스만 덮어쓰고
# 다시 빌드하므로, 이미지가 오래됐더라도 **레포 코드가 항상 우선**한다.
#   https://hub.docker.com/r/kante2/cmu-vln-2026-sysnav
FROM kante2/cmu-vln-2026-sysnav:submission-v5

# system 컨테이너와 동일한 DDS를 쓰도록 고정한다.
#
# 베이스 이미지의 ~/.bashrc도 CycloneDDS를 export하지만 그건 **대화형 셸에서만** 실행된다
# (Ubuntu 기본 .bashrc는 비대화형이면 맨 앞에서 즉시 return). 그래서 `bash -c`나 `bash -lc`로
# 실행하면 기본값인 FastDDS로 떨어진다. 그런데 두 컨테이너는 network_mode: host면서 ipc는
# private이라 /dev/shm이 서로 분리돼 있고, FastDDS는 둘을 "같은 호스트"로 보고 공유메모리
# 전송을 고른다. 그 결과 discovery(UDP 멀티캐스트)는 되는데 데이터 전달만 실패해서,
# `ros2 topic list`에는 토픽이 다 보이는데 메시지는 한 건도 안 오는 상태가 된다. 노드는
# 안 죽고 센서 대기만 하므로 원인을 찾기 어렵다. ENV로 박아두면 실행 방식과 무관하게 보장된다.
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# 레포의 최신 소스로 덮어쓰기. 빌드 컨텍스트는 ai_module/ 이다(docker/compose.yml 기준).
COPY --chown=docker:docker src/sysnav_ros2_mvp /home/docker/ai_module/src/sysnav_ros2_mvp

WORKDIR /home/docker/ai_module
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && \
    colcon build --symlink-install --packages-select sysnav"

# debug 출력 폴더 보장. 베이스 이미지 태그와 무관하게 항상 존재하도록 래퍼에도 둔다.
RUN mkdir -p /home/docker/ai_module/debug/scene_graph_viewpoints

# 평가자용 실행 스크립트.
# docker exec는 ENTRYPOINT(/ros_entrypoint.sh)를 거치지 않고, 비대화형 셸은 ~/.bashrc를
# 읽지 않는다. 그래서 소싱을 여기서 명시적으로 해둔다. 평가자는 아래 한 줄만 실행하면 된다.
#   docker exec -e GEMINI_API_KEY=<key> -it iros2026_ai_module /home/docker/run_sysnav.sh
RUN printf '%s\n' \
      '#!/bin/bash' \
      'set -e' \
      'source /opt/ros/jazzy/setup.bash' \
      'source /home/docker/ai_module/install/setup.bash' \
      'exec ros2 launch sysnav sysnav.launch.py' \
      > /home/docker/run_sysnav.sh && chmod +x /home/docker/run_sysnav.sh
