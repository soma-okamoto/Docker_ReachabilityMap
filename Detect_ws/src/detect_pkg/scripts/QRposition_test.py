#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped


def main():
    rospy.init_node("QRCodeDetection", anonymous=True)

    pub = rospy.Publisher("/QRposition", PoseStamped, queue_size=10)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        msg = PoseStamped()

        # この座標がどの座標系基準かを書く
        # RealSense基準なら camera_link や camera_color_optical_frame など
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "camera_link"

        # テスト用のQRコード位置
        youbot_x = 1.0
        youbot_y = 0.5
        youbot_z = 1.0

        msg.pose.position.x = youbot_x
        msg.pose.position.y = youbot_y
        msg.pose.position.z = youbot_z

        # テスト用の姿勢
        # 回転なし
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        pub.publish(msg)

        rate.sleep()


if __name__ == "__main__":
    main()