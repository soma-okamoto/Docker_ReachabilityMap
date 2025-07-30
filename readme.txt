--------------------------------DockerBuild-----------------
cd Docker_ReachabilityMap
docker build -t ros_dev .


---------------------------------Dockerコンテナ立ち上げ---------------------------------
cd Docker_ReachabilityMap
docker rm -f ros_dev
docker run -d --name ros_dev -p 9090:9090 -v C:\Users\soma0\Docker_ReachabilityMap\RM:/root/RM ros_dev tail -f /dev/null

---------------------------------コンテナエントリー-------------------------------------
cd Docker_ReachabilityMap
docker exec -it ros_dev bash
source /opt/ros/noetic/setup.bash


----------------------------

roslaunch rosbridge_server rosbridge_websocket.launch


----------------ReachabilityMap---------------
cd RM
source devel/setup.bash
roslaunch sampled_reachability_maps MR_IRM_generate_Docker.launch




RosConnector
-------------------------------------
ws://192.168.11.11:9090
ws://localhost:9090
-------------------------------------