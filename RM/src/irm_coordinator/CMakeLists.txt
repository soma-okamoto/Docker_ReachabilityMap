cmake_minimum_required(VERSION 3.0.2)
project(irm_coordinator)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  message_generation
)

add_service_files(
  FILES
  GenerateBaseMap.srv
  GenerateFirstRoute.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    geometry_msgs
    nav_msgs
    sensor_msgs
    message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/coordinator_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
