#!/usr/bin/env python3  # Python 3 で実行することを指定
import rospy  # ROS ノード操作用ライブラリ
import csv  # CSV ファイルへの書き込みに使用
import argparse  # コマンドライン引数解析に使用
from std_msgs.msg import Float32MultiArray  # /bottle_features メッセージ型
import numpy as np  # 配列操作ライブラリ
import os  # パス操作やファイル名抽出に使用
import glob  # ワイルドカードでファイル検索
import re  # 正規表現で番号を抽出
class BottleLogger:
    def __init__(self, csv_path):
        self.seq_id = 0  # シーケンス番号の初期値
         # ボトルIDごとの直前 touch_flag を保持
        self.prev_touches = {}  # 各ボトルの直前タッチ状態を記録する辞書

        self.csv_file = open(csv_path, 'w', newline='')  # CSV ファイルを新規作成して開く
        self.writer = csv.writer(self.csv_file)  # 行書き込み用の writer を作成

        # ヘッダ行に sequence_id を追加
        self.writer.writerow([
            'timestamp','sequence_id','ID','x','y','z',
            'reach_flag','touch_flag',
            's_hand','s_head',s'_accel'
        ])  # CSV のヘッダーを書き込む
        rospy.Subscriber('/bottle_features', Float32MultiArray, self.callback)  # /bottle_features を購読しコールバックに渡す

    def callback(self, msg):
        ts = rospy.get_time()  # ROS 時間でタイムスタンプを取得
        # recs = np.array(msg.data).reshape(-1,10)
        recs = np.array(msg.data).reshape(-1,9)  # メッセージデータを各ボトル分の行に整形
        for rec in recs:
            b_id       = int(rec[0])  # 先頭列からボトルIDを取得
            x, y, z    = rec[1], rec[2], rec[3]  # 位置座標を取り出す
            reach_flag = int(rec[4])  # reach 状態を整数化
            touch_flag = int(rec[5])  # touch 状態を整数化
            # s_hand, s_head, s_accel, s_gaze = rec[6:]
            s_hand, s_head, s_accel= rec[6:]  # 3 つの特徴量を抽出


            # このボトルの直前フラグ
            prev = self.prev_touches.get(b_id, 0)  # 直前のタッチ状態を取得、なければ0
            # ボトルb_idが 1→0 に遷移したときだけシーケンスIDをインクリメント
            if prev == 1 and touch_flag == 0:
                self.seq_id += 1  # タッチ終了を検出したらシーケンス番号を進める
             # フラグ更新
            self.prev_touches[b_id] = touch_flag  # 現在のタッチ状態を記録

            # ここで現在の seq_id を書き込む
            self.writer.writerow([
                ts, self.seq_id, b_id, x, y, z,
                reach_flag, touch_flag,
                s_hand, s_head, s_accel
            ])  # 1 行ぶんの計測値を CSV に追記
            self.csv_file.flush()  # ディスクに即時書き出し



def get_next_csv_path(pattern):

    base, ext = os.path.splitext(pattern)  # 指定パターンをベース名と拡張子に分割
    if ext == '':
        ext = '.csv'  # 拡張子が省略されていた場合は .csv を付与
    # 'base*.csv' にマッチするファイルを列挙
    files = glob.glob(f"{base}*{ext}")  # 既存ファイルをワイルドカードで列挙
    nums = []
    for f in files:
        # ファイル名だけ取り出してマッチ
        name = os.path.basename(f)  # パスからファイル名部だけを抽出
        m = re.match(rf"{re.escape(os.path.basename(base))}(\d+){re.escape(ext)}$", name)
        if m:
            nums.append(int(m.group(1)))  # 接尾番号を整数で収集
    next_num = max(nums) + 1 if nums else 1  # 既存番号の最大＋1、存在しなければ 1
    return f"{base}{next_num}{ext}"  # 次に使うファイルパスを返す



if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # コマンドライン引数パーサを作成
    parser.add_argument('--csv', type=str, default='src/detect_pkg/csv/bottle_logs.csv',
                        help='path to output CSV file')  # 出力先ファイルパターン
    args = parser.parse_args()  # 引数を解析

    # 実行のたびに番号付きファイル名を生成
    csv_path = get_next_csv_path(args.csv)  # 連番付きの実際のファイル名を決定
    rospy.init_node('bottle_logger', anonymous=True)  # bottle_logger ノードを初期化
    logger = BottleLogger(csv_path)  # ロガーインスタンスを生成し購読を開始
    rospy.loginfo(f"Logging to {csv_path}")  # 保存先をログ出力
    rospy.spin()  # ノードを回してコールバックを待機
    logger.csv_file.close()  # ノード終了時にファイルを閉じる
