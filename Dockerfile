FROM ros:noetic-ros-core

# タイムゾーンを JST に設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ > /etc/timezone

# 必要な apt パッケージのインストール
RUN apt-get update && apt-get install -y \
      ros-noetic-rosbridge-server \
      python3-pip \
      python3-catkin-tools \
      python3-rosdep \
      ros-noetic-rviz \
      qtbase5-dev \
      qtbase5-dev-tools \
      libqt5core5a \
      libqt5gui5 \
      libqt5widgets5 \
      libhdf5-dev \
      dos2unix \
    && rm -rf /var/lib/apt/lists/*



# Python モジュールを pip 経由でインストール
RUN pip3 install --no-cache-dir \
      "numpy>=1.20,<2.0" \
      scipy \
      h5py \
      joblib \
      matplotlib \
      torch \
      torchvision \
      pytorch-kinematics

# rosdep 初期化
RUN rosdep init \
 && rosdep update


# 作業ディレクトリ設定
WORKDIR /root

# デフォルトシェル
CMD ["bash"]
