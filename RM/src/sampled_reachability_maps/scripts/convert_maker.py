#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker

def irm_callback(msg):
    data = msg.data
    count = len(data) // 4

    marker = Marker()
    marker.header.frame_id = "base_footprint"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "irm_map"
    marker.id = 0
    marker.type = Marker.SPHERE_LIST  # または POINTS でもOK
    marker.action = Marker.ADD
    marker.scale.x = 0.03  # 1点あたりの球体直径
    marker.scale.y = 0.03
    marker.scale.z = 0.03
    marker.color.a = 1.0
    marker.lifetime = rospy.Duration(0)  # 永続

    # 色つきにしたい場合は marker.colors も使う
    marker.points = []
    for i in range(count):
        from geometry_msgs.msg import Point
        p = Point()
        p.x = data[i * 4 + 0]
        p.y = data[i * 4 + 1]
        p.z = data[i * 4 + 2]
        marker.points.append(p)
        # 必要に応じて色も追加

    irm_marker_pub.publish(marker)

if __name__ == "__main__":
    rospy.init_node("irm_map_marker_viz")
    irm_marker_pub = rospy.Publisher("/IRM_Map_marker", Marker, queue_size=1)
    rospy.Subscriber("/IRM_Map", Float32MultiArray, irm_callback)
    rospy.loginfo("IRM_Map -> Marker変換ノード起動")
    rospy.spin()
