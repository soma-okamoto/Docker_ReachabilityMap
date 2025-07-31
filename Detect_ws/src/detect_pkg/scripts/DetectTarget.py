#!/usr/bin/env python3
import rospy
import json
import argparse
from std_msgs.msg import Int32, Float32MultiArray
import numpy as np
from joblib import load

class BottleIdentifier:
    def __init__(self, win, wout, tin, tout, alpha, hysteresis_m,
                 n_fields=None):
        if n_fields is None:
            raise ValueError("n_fields must be provided")
        self.n_fields = int(n_fields)

        self.win  = np.array(win,  dtype=np.float32)
        self.wout = np.array(wout, dtype=np.float32)

        self.tin   = tin
        self.tout  = tout
        self.alpha = alpha
        self.M     = hysteresis_m

        self.smoothed = {}
        self.prev_best = None
        self.counter   = 0

        self.pub = rospy.Publisher('/identified_bottle', Int32, queue_size=10)
        # self.pub = rospy.Publisher('/identified_bottle', Float32MultiArray, queue_size=10)
        rospy.Subscriber('/bottle_features', Float32MultiArray, self.callback)


    def callback(self, msg: Float32MultiArray):


        data = np.array(msg.data, dtype=np.float32)
        try:
            recs = data.reshape(-1, self.n_fields)
            num_scores = self.n_fields - 6
        except ValueError:
            rospy.logwarn("Unexpected data length %d, cannot reshape to (-1,%d)",
                          len(data), self.n_fields)
            return

        # まず各ボトルの位置・reach_flag を取得
        reach_flags = {}
        b_positions = {}
        touch_flags = {}
        w_scores = {}

        for rec in recs:
            b_id, pos, rf ,tf = int(rec[0]), rec[1:4], int(rec[4]),int(rec[5])
            reach_flags[b_id] = rf
            touch_flags[b_id]=tf
            b_positions[b_id] = tuple(pos.tolist())
            scores_np = rec[6:6+num_scores].astype(np.float32)
            weights = self.win if reach_flags[b_id] == 1 else self.wout

            if weights.size != scores_np.size:
                rospy.logwarn(
                    "Weight length %d != feature length %d, skipping b_id=%d",
                    weights.size, scores_np.size, b_id
                )
                continue

            w_scores[b_id] = float(weights.dot(scores_np))
        
        # EMA 前の古いキー削除
        for old in list(self.smoothed):
            if old not in w_scores:
                del self.smoothed[old]

        # EMA 平滑化
        for b_id, score in w_scores.items():
            prev = self.smoothed.get(b_id, score)
            self.smoothed[b_id] = self.alpha * score + (1 - self.alpha) * prev
        
        # TouchFlagがあるかチェック
        touched_ids = [b_id for b_id, tf in touch_flags.items() if tf == 1]
        if touched_ids:
            best_id = touched_ids[0]
            best_score = self.smoothed.get(best_id, 0.0)
            pos = b_positions[best_id]
            rospy.loginfo(f"Touch bottle: ID={best_id} , {pos}, score={best_score:.3f}")
            self.pub.publish(best_id)
            self.prev_best, self.counter = best_id, self.M
            return

        # 最有力候補判定＋ヒステリシス
        best_id, best_score = max(self.smoothed.items(), key=lambda x: x[1])
        threshold = self.tin if reach_flags.get(best_id, False) else self.tout

        if best_score >= threshold:
            if best_id == self.prev_best:
                self.counter += 1
            else:
                self.prev_best, self.counter = best_id, 1
        else:
            # しきい値を下回ったら即座に候補なしを通知
            if self.prev_best is not None:
                rospy.loginfo("No candidate satisfies threshold. Resetting.")
                self.pub.publish(-1)
            self.prev_best, self.counter = None, 0
            return  # ここで処理を終了

        if self.counter >= self.M:
            pos = b_positions[best_id]
            reach=reach_flags[best_id]
            rospy.loginfo(f"Detect bottle: ID={best_id} , Reach={reach},{pos}, score={best_score:.3f}")
            self.pub.publish(best_id)
            # self.pub.publish(best_id,reach)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Bottle identification'
    )
    parser.add_argument('--mode',      choices=['load','set'], default='set')
    parser.add_argument('--weights',   type=str)
    parser.add_argument('--win',       type=str)
    parser.add_argument('--wout',      type=str)
    parser.add_argument('--tin',       type=float, default=0.5)
    parser.add_argument('--tout',      type=float, default=0.5)
    parser.add_argument('--alpha',     type=float, default=0.45)
    parser.add_argument('--hysteresis_m',type=int,   default=3)
    parser.add_argument('--n_fields',  type=int, default=9)
    args = parser.parse_args()

 
    # 手動重みモード
    if args.mode == 'load':
        cfg  = json.load(open(args.weights))
        win  = cfg['W_in'];   wout = cfg['W_out']
        tin  = cfg['T_in'];   tout = cfg['T_out']
    else:
        win  = list(map(float, args.win.split(',')))
        wout = list(map(float, args.wout.split(',')))
        tin  = args.tin;     tout = args.tout

    rospy.init_node('bottle_identifier')
    BottleIdentifier(
        win, wout, tin, tout,
        args.alpha, args.hysteresis_m,
        n_fields=args.n_fields
    )
    rospy.loginfo("Waiting for User Signal ...")
    rospy.spin()
