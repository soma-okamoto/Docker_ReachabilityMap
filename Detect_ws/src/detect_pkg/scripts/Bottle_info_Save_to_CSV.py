#!/usr/bin/env python3
import rospy
import csv
import argparse
from std_msgs.msg import Float32MultiArray
import numpy as np
import os
import glob
import re
class BottleLogger:
    def __init__(self, csv_path):
        self.seq_id = 0
         # ボトルIDごとの直前 touch_flag を保持
        self.prev_touches = {}

        self.csv_file = open(csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)

        # ヘッダ行に sequence_id を追加
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


            # このボトルの直前フラグ
            prev = self.prev_touches.get(b_id, 0)
            # ボトルb_idが 1→0 に遷移したときだけシーケンスIDをインクリメント
            if prev == 1 and touch_flag == 0:
                self.seq_id += 1
             # フラグ更新
            self.prev_touches[b_id] = touch_flag

            # ここで現在の seq_id を書き込む
            self.writer.writerow([
                ts, self.seq_id, b_id, x, y, z,
                reach_flag, touch_flag,
                s_hand, s_head, s_accel
            ])
            self.csv_file.flush()



def get_next_csv_path(pattern):
    """
    pattern: --csv で渡されたベースファイル名（たとえば '.../bottle_logs.csv'）
    既存の同ディレクトリ内のファイルを glob で探し、
    'bottle_logs1.csv','bottle_logs2.csv',… のうち最大の数字を取得して +1 する。
    """
    base, ext = os.path.splitext(pattern)
    if ext == '':
        ext = '.csv'
    # 'base*.csv' にマッチするファイルを列挙
    files = glob.glob(f"{base}*{ext}")
    nums = []
    for f in files:
        # ファイル名だけ取り出してマッチ
        name = os.path.basename(f)
        m = re.match(rf"{re.escape(os.path.basename(base))}(\d+){re.escape(ext)}$", name)
        if m:
            nums.append(int(m.group(1)))
    next_num = max(nums) + 1 if nums else 1
    return f"{base}{next_num}{ext}"



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='src/detect_pkg/csv/bottle_logs.csv',
                        help='path to output CSV file')
    args = parser.parse_args()

    # 実行のたびに番号付きファイル名を生成
    csv_path = get_next_csv_path(args.csv)
    rospy.init_node('bottle_logger', anonymous=True)
    logger = BottleLogger(csv_path)
    rospy.loginfo(f"Logging to {csv_path}")
    rospy.spin()
    logger.csv_file.close()
