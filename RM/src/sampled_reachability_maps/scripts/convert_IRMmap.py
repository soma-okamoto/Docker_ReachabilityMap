#!/usr/bin/env python3
import rospy
from reachability_map_visualizer.msg import WorkSpace
from std_msgs.msg import Float32MultiArray

def callback(msg):
    # WorkSpace型 -> Float32MultiArrayへの変換
    out = Float32MultiArray()
    data = []
    for sphere in msg.WsSpheres:
        x = sphere.point.x
        y = sphere.point.y
        z = sphere.point.z
        score = sphere.ri
        data.extend([x, y, z, score])
    out.data = data
    irm_pub.publish(out)
    rospy.loginfo(f"Published {len(msg.WsSpheres)} spheres to /IRM_map")

if __name__ == '__main__':
    rospy.init_node("workspace_to_irmmap_converter")
    irm_pub = rospy.Publisher("/IRM_map", Float32MultiArray, queue_size=1)
    rospy.Subscriber("/reachability_map", WorkSpace, callback)
    rospy.loginfo("Waiting for /reachability_map ...")
    rospy.spin()
