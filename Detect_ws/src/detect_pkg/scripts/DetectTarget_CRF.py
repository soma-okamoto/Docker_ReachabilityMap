#!/usr/bin/env python3
import rospy
import json
import argparse
from std_msgs.msg import Int32, Float32MultiArray
import numpy as np
from joblib import load

class BottleIdentifier:
    def __init__(self, win, wout, tin, tout, alpha, hysteresis_m,
                 model_path=None, n_fields=None, window=None):
        if n_fields is None:
            raise ValueError("n_fields must be provided")
        self.n_fields = int(n_fields)
        self.window   = window or 30  # デフォルトで直近30フレームを保持

        # モデルロード: CRF か ロジスティック回帰か
        if model_path:
            self.model = load(model_path)
            # CRFなら特徴バッファを各ボトルIDに持つ
            if hasattr(self.model, 'state_features_'):
                self.is_crf = True
                self.buffers = {}
                  # CRF 用ヒステリシス用ステートを追加
                self.prev_pred   = None
                self.pred_counter = 0
            else:
                self.is_crf = False
                self.classes_ = list(self.model.classes_)
        else:
            self.is_crf = False
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
        for rec in recs:
            b_id, pos, rf = int(rec[0]), rec[1:4], int(rec[4])
            reach_flags[b_id] = rf
            b_positions[b_id] = tuple(pos.tolist())

        # CRF モードなら一度だけまとめて予測
        if self.is_crf:
            # 全ボトル分の特徴辞書を作成
            feat = {}
            for rec in recs:
                b_id = int(rec[0])
                flag       = int(rec[4])    # reach_flag
                s_hand, s_head, s_accel = rec[6:9]
                # s_hand, s_head, s_accel,s_gaze = rec[6:10]
                feat[f's_hand_in_{b_id}']  = float(s_hand) if flag==1 else 0.0
                feat[f's_hand_out_{b_id}']  = float(s_hand)   if flag==0 else 0.0            
                feat[f's_head_in_{b_id}']  = float(s_head)if flag==1 else 0.0
                feat[f's_head_out_{b_id}']  = float(s_head)if flag==0 else 0.0
                feat[f's_accel_in_{b_id}'] = float(s_accel)if flag==1 else 0.0
                feat[f's_accel_out_{b_id}'] = float(s_accel)if flag==0 else 0.0
                # feat[f's_gaze_in_{b_id}'] = float(s_gaze)if flag==1 else 0.0
                # feat[f's_gaze_out_{b_id}'] = float(s_gaze)if flag==0 else 0.0

            # バッファに追加して長さを制限
            buf = self.buffers.setdefault('all', [])
            buf.append(feat)
            if len(buf) > self.window:
                buf.pop(0)
            

            # ③ Viterbi
            if len(buf) < self.window:
                return

                # デバッグ出力
            if len(buf) == self.window:

                # ここで初めて predict
                y_seq = self.model.predict([buf])[0]
                rospy.loginfo(f"[DEBUG] predicted sequence = {y_seq}")
                pred = int(y_seq[-1])
                rospy.loginfo(f"[CRF] publishing → {pred}")
                self.pub.publish(pred)
                return


        # # 非 CRF モードなら従来の重み＋EMA＋ヒステリシス処理
        w_scores = {}
        for rec in recs:
            b_id = int(rec[0])
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

        # 最有力候補判定＋ヒステリシス
        best_id, best_score = max(self.smoothed.items(), key=lambda x: x[1])
        threshold = self.tin if reach_flags.get(best_id, False) else self.tout

        if best_score >= threshold:
            if best_id == self.prev_best:
                self.counter += 1
            else:
                self.prev_best, self.counter = best_id, 1
        else:
            self.prev_best, self.counter = None, 0

        if self.counter >= self.M:
            pos = b_positions[best_id]
            rospy.loginfo(f"Identified bottle {best_id} at {pos}, score={best_score:.3f}")
            self.pub.publish(best_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Bottle identification with optional CRF'
    )
    parser.add_argument('--mode',      choices=['load','set'], default='set')
    parser.add_argument('--weights',   type=str)
    parser.add_argument('--win',       type=str)
    parser.add_argument('--wout',      type=str)
    parser.add_argument('--tin',       type=float, default=0.5)
    parser.add_argument('--tout',      type=float, default=0.5)
    parser.add_argument('--alpha',     type=float, default=0.5)
    parser.add_argument('--hysteresis_m',type=int,   default=3)
    # parser.add_argument('--model',     type=str,   default="/home/dars/Detect_ws/src/detect_pkg/Train_joblib/bottle_crf_model.joblib",  help='bottle_crf_model.joblib model path')
    parser.add_argument('--model',     type=str,  help='bottle_crf_model.joblib model path')
    
    parser.add_argument('--n_fields',  type=int, default=9,  required=True)
    parser.add_argument('--window',    type=int,   default=30,
                        help='CRF sequence window length')
    args = parser.parse_args()

    # --- CRF モードなら win/wout は不要 ---
    if args.model:
        win = wout = None
        tin = tout = None
    else:
        # 手動重みモード
        if args.mode == 'load':
            cfg  = json.load(open(args.weights))
            win  = cfg['W_in'];   wout = cfg['W_out']
            tin  = cfg['T_in'];   tout = cfg['T_out']
        else:
            # --win, --wout が None のときはエラー
            if args.win is None or args.wout is None:
                parser.error('--win and --wout are required when not using --model')
            win  = list(map(float, args.win.split(',')))
            wout = list(map(float, args.wout.split(',')))
            tin  = args.tin;     tout = args.tout

    rospy.init_node('bottle_identifier')
    BottleIdentifier(
        win, wout, tin, tout,
        args.alpha, args.hysteresis_m,
        model_path=args.model,
        n_fields=args.n_fields,
        window=args.window
    )
    rospy.spin()
