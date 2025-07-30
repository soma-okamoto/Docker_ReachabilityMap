<launch>
  <!-- 引数定義 -->
  <arg name="map_file_1" default="/home/dasnote11/catkin_ws/src/sampled_reachability_maps/data/3D_base_reach_map_gripper_finger_link_torso_True_0.08_2024-12-20-23-07-48.h5"/>
  <arg name="map_file_2" default="/home/dasnote11/catkin_ws/src/sampled_reachability_maps/data/3_pkl_reach_map_gripper_finger_link_torso_True_0.08_2024-12-20-23-07-48.h5"/>

  <!-- ノード定義 -->
  <node name="load_map_1" pkg="reachability_map_visualizer" type="load_reachability_map" output="screen">
    <param name="map_file" value="$(arg map_file_1)" />
  </node>

  <node name="load_map_2" pkg="reachability_map_visualizer" type="load_reachability_map1" output="screen">
    <param name="map_file" value="$(arg map_file_2)" />
  </node>
</launch>
