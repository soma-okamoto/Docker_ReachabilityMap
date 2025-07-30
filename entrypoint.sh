#!/bin/bash
set -e

# ROS環境
source /opt/ros/noetic/setup.bash

# 1) roscore をバックグラウンドで立ち上げ
roscore &
# 少し待って ROS Master を立ち上げ
sleep 5

# 2) rosbridge_websocket をフォアグラウンドで起動
exec roslaunch rosbridge_server rosbridge_websocket.launch
