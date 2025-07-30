#include <ros/ros.h>
#include "reachability_map_visualizer/WorkSpace.h"
#include <reachability_map_visualizer/hdf5_dataset.h>

int main(int argc, char **argv)
{
  if (argc < 2) //
  {
    ROS_ERROR_STREAM("Please provide the name of the reachability map. If you have not created it yet, Please create "
                     "the map by running the create reachability map launch file in reachability_map_visualizer");
    return 1;
  }
  else
  {
    ros::init(argc, argv, "workspace");
    ros::NodeHandle n;

    // TODO: It can be published as a latched topic. So the whole message will be published just once and stay on the
    // topic
    ros::Publisher workspace_pub = n.advertise< reachability_map_visualizer::WorkSpace >("reachability_map", 1);
    // bool latchOn = 1; //reachability_mapというトピック名で,一つのメッセージを送信している.
    // ros::Publisher workspace_pub = n.advertise<reachability_map_visualizer::WorkSpace>("reachability_map", 1, latchOn);
    ros::Rate loop_rate(10);

    int count = 0;

    hdf5_dataset::Hdf5Dataset h5(argv[1]); //リー茶ビリティマップのデータセットを読み込んでいる
    h5.open();

    MultiMapPtr pose_col_filter;
    MapVecDoublePtr sphere_col;
    float res;
    h5.loadMapsFromDataset(pose_col_filter, sphere_col, res);

    ROS_INFO("Map loading complete");
    // Creating messages
    reachability_map_visualizer::WorkSpace ws;
    ws.header.stamp = ros::Time::now();
    ws.header.frame_id = "base_footprint";
    ws.resolution = res;

    for (MapVecDoublePtr::iterator it = sphere_col.begin(); it != sphere_col.end(); ++it)
    {
       reachability_map_visualizer::WsSphere wss;
       wss.point.x = (*it->first)[0];
       wss.point.y = (*it->first)[1];
       wss.point.z = (*it->first)[2];
       wss.ri = it->second;

       for (MultiMapPtr::iterator it1 = pose_col_filter.lower_bound(it->first); it1 != pose_col_filter.upper_bound(it->first); ++it1)
       {
          geometry_msgs::Pose pp;
          pp.position.x = it1->second->at(0);
          pp.position.y = it1->second->at(1);
          pp.position.z = it1->second->at(2);
          pp.orientation.x = it1->second->at(3);
          pp.orientation.y = it1->second->at(4);
          pp.orientation.z = it1->second->at(5);
          pp.orientation.w = it1->second->at(6);
          wss.poses.push_back(pp);
        }
        ws.WsSpheres.push_back(wss);
      }

    while (ros::ok())
    {
      workspace_pub.publish(ws);

      ros::spinOnce();
      sleep(5);
      ++count;
    }
  }
  return 0;
}
