#!/usr/bin/env python3
import rospy
import csv
import argparse
from std_msgs.msg import Float32MultiArray
import numpy as np

class BottleLogger:
    def __init__(self, csv_path):
        self.seq_id = 0
        self.prev_touch = 0
        self.csv_file = open(csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        # ヘッダ行に sequence_id を追加
        # self.writer.writerow([
        #     'timestamp','sequence_id','ID','x','y','z',
        #     'reach_flag','touch_flag',
        #     's_hand','s_head','s_accel','s_gaze'
        # ])

        self.writer.writerow([
            'timestamp','sequence_id','ID','x','y','z',
            'reach_flag','touch_flag',
            's_hand','s_head','s_accel'
        ])
        rospy.Subscriber('/bottle_features', Float32MultiArray, self.callback)

    def callback(self, msg):
        ts = rospy.get_time()
        # recs = np.array(msg.data).reshape(-1,10)
        recs = np.array(msg.data).reshape(-1,9)
        for rec in recs:
            b_id       = int(rec[0])
            x, y, z    = rec[1], rec[2], rec[3]
            reach_flag = int(rec[4])
            touch_flag = int(rec[5])
            # s_hand, s_head, s_accel, s_gaze = rec[6:]
            s_hand, s_head, s_accel= rec[6:]

            # ここで現在の seq_id を書き込む
            # self.writer.writerow([
            #     ts, self.seq_id, b_id, x, y, z,
            #     reach_flag, touch_flag,
            #     s_hand, s_head, s_accel, s_gaze
            # ])
            self.writer.writerow([
                ts, self.seq_id, b_id, x, y, z,
                reach_flag, touch_flag,
                s_hand, s_head, s_accel
            ])
            self.csv_file.flush()

            # touch_flag の 1→0 の遷移時にシーケンスIDをインクリメント
        # ここでは前フレームの touch_flag を保持し、
        # 現在のフラグが 0 かつ前のフラグが 1 の場合にシーケンスを切り替える
            if self.prev_touch == 1 and touch_flag == 0:
                self.seq_id += 1
            self.prev_touch = touch_flag


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='src/detect_pkg/csv/bottle_logs.csv',
                        help='path to output CSV file')
    args = parser.parse_args()

    rospy.init_node('bottle_logger', anonymous=True)
    logger = BottleLogger(args.csv)
    rospy.spin()
    # ノード終了時にファイルを閉じる
    logger.csv_file.close()
