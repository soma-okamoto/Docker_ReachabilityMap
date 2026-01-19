FROM ros:noetic-ros-core

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ > /etc/timezone

# 依存を「明示」して再現性を上げる
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip \
      python3-catkin-tools \
      python3-rosdep \
      libnss-wrapper \
      libhdf5-dev \
      dos2unix \
      # --- ROSの基本ツール＆Python(RM/Detectに必須) ---
      ros-noetic-rospy \
      ros-noetic-rviz \
      ros-noetic-roslaunch \
      ros-noetic-rostopic \
      ros-noetic-rospack \
      ros-noetic-std-msgs \
      ros-noetic-sensor-msgs \
      ros-noetic-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# pip 基盤
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Python依存（旧Dockerfileに合わせて維持）
RUN python3 -m pip install --no-cache-dir \
      "numpy>=1.20,<2.0" \
      scipy \
      h5py \
      joblib \
      matplotlib \
      torch \
      torchvision \
      pytorch-kinematics

# rosdep（環境によって既に初期化済みの場合があるので安全化）
RUN rosdep init 2>/dev/null || true
RUN rosdep update || true


# bash起動時にROS環境を読み込み（便利）
RUN echo "source /opt/ros/noetic/setup.bash" >> /etc/bash.bashrc

# 非rootでも I have no name! を消す entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /work
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
