#!/usr/bin/env python3
import rospy
import csv
import os
from reachability_map_visualizer.msg import WorkSpace

def callback(msg):
    rospy.loginfo("Received reachability map, saving to CSV...")

    # CSVファイルパス
    csv_file = os.path.expanduser('~/reachability_map.csv')

    # CSVファイルを開く
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # CSVヘッダーを書き込む
        writer.writerow(['x', 'y', 'z', 'score'])

        # すべての球体をループ
        for sphere in msg.WsSpheres:
            x = sphere.point.x
            y = sphere.point.y
            z = sphere.point.z
            score = sphere.ri

            # CSVに書き込む
            writer.writerow([x, y, z, score])

    rospy.loginfo(f"Reachability map saved to {csv_file}")

    # 一回保存したら終了
    rospy.signal_shutdown("Saved CSV")


def listener():
    rospy.init_node('reachability_csv_saver', anonymous=True)
    rospy.Subscriber('/reachability_map', WorkSpace, callback)
    rospy.loginfo("Waiting for /reachability_map message...")
    rospy.spin()


if __name__ == '__main__':
    listener()