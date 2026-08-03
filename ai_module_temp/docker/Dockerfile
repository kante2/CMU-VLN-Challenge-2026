FROM zhangjicmu/ubuntu24_ros:ai_module

# Copy updated dummy_vlm source into the existing workspace
COPY src/dummy_vlm /home/docker/ai_module/src/dummy_vlm

# Rebuild only dummy_vlm against the already-built workspace dependencies
WORKDIR /home/docker/ai_module
RUN /bin/bash -c "\
    source /opt/ros/jazzy/setup.bash && \
    colcon build \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-select dummy_vlm"

# Ensure the updated install is sourced in every new shell
RUN echo "source /home/docker/ai_module/install/setup.bash" >> /home/docker/.bashrc
