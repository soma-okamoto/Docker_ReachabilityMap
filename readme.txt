--------------------------------DockerBuild-----------------
cd Docker_ReachabilityMap
docker build -t ros_dev .


---------------------------------Dockerコンテナ立ち上げ---------------------------------
cd Docker_ReachabilityMap
docker rm -f ros_dev
docker run -d --name ros_dev -p 9090:9090 -v C:\Users\soma0\Docker_ReachabilityMap:/root ros_dev tail -f /dev/null

---------------------------------コンテナエントリー-------------------------------------
cd Docker_ReachabilityMap
docker exec -it ros_dev bash

source /opt/ros/noetic/setup.bash

----------------------------
source /opt/ros/noetic/setup.bash

roslaunch rosbridge_server rosbridge_websocket.launch


----------------ReachabilityMap---------------
cd RM
source devel/setup.bash

export ROS_MASTER_URI=http://192.168.11.11:11311
export ROS_IP=host.docker.internal

roslaunch sampled_reachability_maps MR_IRM_generate_Docker.launch

rosrun sampled_reachability_maps MR_IRM_firstRoute.py
rosrun sampled_reachability_maps MR_IRM_firstRoute_fixed.py

cd Detect_ws
source devel/setup.bash

export ROS_MASTER_URI=http://192.168.11.11:11311
export ROS_IP=host.docker.internal

rosrun detect_pkg DetectTarget.py \
  --win=0.5,0.25,0.25 \
  --wout=0.21,0.58,0.21

RosConnector
-------------------------------------
ws://192.168.11.11:9090
ws://localhost:9090
-------------------------------------