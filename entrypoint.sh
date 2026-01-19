#!/bin/bash
set -e

# ROS環境（rospy/roslaunch/rostopic が安定）
source /opt/ros/noetic/setup.bash

# 非root運用向け
export HOME="${HOME:-/tmp}"

# ここでは passwd いじり系はやらない（docker exec に効かないため）
# whoami を確実に直すのは /usr/local/bin/enter 側でやる

exec "$@"
