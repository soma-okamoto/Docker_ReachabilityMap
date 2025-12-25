#!/usr/bin/env python3
import rospy
import csv
import os
from std_msgs.msg import Float32MultiArray

def callback(msg):
    rospy.loginfo("Inverse_Received reachability map, saving to CSV...")

    csv_file = os.path.expanduser('~/Inverse_reachability_map.csv')

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'z', 'score'])

        # Float32MultiArrayのデータは[x,y,z,score,x,y,z,score,...]の並び
        data = msg.data
        for i in range(0, len(data), 4):
            x, y, z, score = data[i:i+4]
            writer.writerow([x, y, z, score])

    rospy.loginfo(f"Inverse Reachability map saved to {csv_file}")

    rospy.signal_shutdown("Saved CSV")

def listener():
    rospy.init_node('Inverse_reachability_csv_saver', anonymous=True)
    rospy.Subscriber('/IRM_Map', Float32MultiArray, callback)
    rospy.loginfo("Waiting for /IRM_Map message...")
    rospy.spin()

if __name__ == '__main__':
    listener()
