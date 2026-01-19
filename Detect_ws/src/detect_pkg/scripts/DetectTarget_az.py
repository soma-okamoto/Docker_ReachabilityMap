#!/usr/bin/env python3 
# シバンで Python 3 を指定
import rospy  # ROS ノード制御用ライブラリ
import json  # 重み設定の読み書きに使用
import argparse  # コマンドライン引数のパーサ
from std_msgs.msg import Int32, Float32,Float32MultiArray  # ROS メッセージ型をインポート
import numpy as np  # 数値計算ライブラリ
from joblib import load  # joblib 由来のロード関数（現在は未使用だが保持）
from std_msgs.msg import Bool  # Bool 型メッセージ（コメントアウト箇所で使用予定）

class BottleIdentifier:
    def __init__(self, win, wout, tin, tout, alpha, hysteresis_m,
                 n_fields=None):
        if n_fields is None:
            raise ValueError("n_fields must be provided")  # 必須引数がなければ例外
        self.n_fields = int(n_fields)  # 受信レコードのフィールド数を保存

        self.win  = np.array(win,  dtype=np.float32)  # リーチ内用の重みを float32 配列に変換
        self.wout = np.array(wout, dtype=np.float32)  # リーチ外用の重みを float32 配列に変換

        self.tin   = tin  # リーチ内の採択しきい値
        self.tout  = tout  # リーチ外の採択しきい値
        self.alpha = alpha  # EMA の平滑化係数
        self.M     = hysteresis_m  # ヒステリシスで必要な連続カウント

        self.smoothed = {}  # ボトル毎の平滑化スコアを保持
        self.prev_best = None  # 前回確定候補の ID
        self.counter   = 0  # ヒステリシス用の連続カウント

        self.pub = rospy.Publisher('/identified_bottle', Int32, queue_size=10)  # 採択 ID を配信
        self.pub_score = rospy.Publisher('/identified_bottle_score', Float32, queue_size=10)  # スコアを配信
        rospy.Subscriber('/bottle_features', Float32MultiArray, self.callback)  # 特徴量トピックを購読
        self.pub_pos  = rospy.Publisher('/identified_bottle_pose', Float32MultiArray, queue_size=10)  # ボトル位置を配信
        # self.pub_touch = rospy.Publisher('/identified_bottle_touch', Bool,   queue_size=10)  # Bool でタッチ通知する場合の定義
        self.pub_touch = rospy.Publisher('/identified_bottle_touch', Int32,   queue_size=10)  # Int32 でタッチ通知を配信


    def callback(self, msg: Float32MultiArray):


        data = np.array(msg.data, dtype=np.float32)  # 配列化して float32 に揃える
        try:
            recs = data.reshape(-1, self.n_fields)  # 1 次元配列を (ボトル数×フィールド数) に再整形
            num_scores = self.n_fields - 6  # 先頭6項目を除いたスコア列長
        except ValueError:
            rospy.logwarn("Unexpected data length %d, cannot reshape to (-1,%d)",
                          len(data), self.n_fields)  # 形状が合わなければ警告
            return  # 処理を中止

        # まず各ボトルの位置・reach_flag を取得
        reach_flags = {}  # ボトル毎の reach フラグを保持
        b_positions = {}  # ボトル毎の座標を保持
        touch_flags = {}  # ボトル毎のタッチ状態を保持
        w_scores = {}  # 加重スコアを格納

        for rec in recs:
            b_id, pos, rf ,tf = int(rec[0]), rec[1:4], int(rec[4]),int(rec[5])  # 1 レコードから ID・位置・フラグを抽出
            reach_flags[b_id] = rf  # reach フラグを辞書に保存
            touch_flags[b_id]=tf  # タッチフラグを辞書に保存
            b_positions[b_id] = tuple(pos.tolist())  # numpy 配列をタプルに変換して保持
            scores_np = rec[6:6+num_scores].astype(np.float32)  # 特徴量スコア部分を取り出す
            weights = self.win if reach_flags[b_id] == 1 else self.wout  # reach 内外で重みを切り替え

            if weights.size != scores_np.size:
                rospy.logwarn(
                    "Weight length %d != feature length %d, skipping b_id=%d",
                    weights.size, scores_np.size, b_id
                )  # 特徴量数と重み数が一致しなければ警告
                continue  # このボトルをスキップ

            w_scores[b_id] = float(weights.dot(scores_np))  # 内積を取ってスコア化
        
        # EMA 前の古いキー削除
        for old in list(self.smoothed):
            if old not in w_scores:
                del self.smoothed[old]  # 現在観測されないボトルは平滑化辞書から除外

        # EMA 平滑化
        for b_id, score in w_scores.items():
            prev = self.smoothed.get(b_id, score)  # 過去スコアが無ければ現在値を使用
            self.smoothed[b_id] = self.alpha * score + (1 - self.alpha) * prev  # EMA を計算
        
        # TouchFlagがあるかチェック
        touched_ids = [b_id for b_id, tf in touch_flags.items() if tf == 1]  # タッチ中のボトル一覧を作成
        if touched_ids:
            best_id = touched_ids[0]  # 最初のタッチ対象を優先
            best_score = self.smoothed.get(best_id, 0.0)  # 平滑化スコアを取得
            pos = b_positions[best_id]  # 位置を取得
            rospy.loginfo(f"Touch bottle: ID={best_id} , {pos}, score={best_score:.3f}")  # ログ出力
            self.pub.publish(best_id)  # ボトル ID を publish
            self.pub_pos.publish(Float32MultiArray(data=pos))  # 位置を publish
            self.pub_score.publish(best_score)  # スコアを publish
            if touch_flags[best_id]:
                touch=1  # タッチしている場合は 1
            else:
                touch=0  # タッチしていない場合は 0
            self.pub_touch.publish(touch)  # タッチ状態を publish
            self.prev_best, self.counter = best_id, self.M  # ヒステリシス状態を更新して確定
            return  # タッチが優先なので処理終了

        # 最有力候補判定＋ヒステリシス
        best_id, best_score = max(self.smoothed.items(), key=lambda x: x[1])  # スコア最大のボトルを取得
        threshold = self.tin if reach_flags.get(best_id, False) else self.tout  # reach 状態に応じてしきい値を選択

        if best_score >= threshold:
            if best_id == self.prev_best:
                self.counter += 1  # 同一候補が継続ならカウンタを増やす
            else:
                self.prev_best, self.counter = best_id, 1  # 候補が変わればカウンタをリセット
        else:
            # しきい値を下回ったら即座に候補なしを通知
            if self.prev_best is not None:
                rospy.loginfo("No candidate satisfies threshold. Resetting.")  # 状態解消を通知
                self.pub.publish(-1)  # ボトルなしを示す -1 を publish
            self.prev_best, self.counter = None, 0  # 状態をリセット
            return  # ここで処理を終了

        if self.counter >= self.M:
            pos = b_positions[best_id]  # ベスト候補の位置を取得
            reach=reach_flags[best_id]  # reach フラグを参照
            rospy.loginfo(f"Detect bottle: ID={best_id} , Reach={reach},{pos}, score={best_score:.3f}")  # 採択をログ
            self.pub.publish(best_id)  # ID を publish
            self.pub_score.publish(best_score)  # スコアを publish
            
            # self.pub.publish(best_id,reach)  # reach と合わせて送る場合の旧コード

            self.pub_pos.publish(Float32MultiArray(data=pos))  # 位置を publish
            if touch_flags[best_id]:
                touch=1  # タッチ状態を数値化
            else:
                touch=0  # 非タッチの場合
            self.pub_touch.publish(touch)  # タッチ情報を publish

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Bottle identification'
    )  # スクリプト説明付きで ArgumentParser を生成
    parser.add_argument('--mode',      choices=['load','set'], default='set')  # 重みの読み込みモード
    parser.add_argument('--weights',   type=str)  # JSON 重みファイルパス
    parser.add_argument('--win',       type=str)  # reach 内重みをカンマ区切りで受け取る
    parser.add_argument('--wout',      type=str)  # reach 外重みをカンマ区切りで受け取る
    parser.add_argument('--tin',       type=float, default=0.5)  # reach 内しきい値
    parser.add_argument('--tout',      type=float, default=0.5)  # reach 外しきい値
    parser.add_argument('--alpha',     type=float, default=0.45)  # EMA 係数
    parser.add_argument('--hysteresis_m',type=int,   default=3)  # ヒステリシスの連続必要回数
    parser.add_argument('--n_fields',  type=int, default=9)  # 1 レコードのフィールド数
    args = parser.parse_args()  # 引数をパース

 
    # 手動重みモード
    if args.mode == 'load':
        cfg  = json.load(open(args.weights))  # JSON から重みセットを読み込む
        win  = cfg['W_in'];   wout = cfg['W_out']  # reach 内外の重みを抽出
        tin  = cfg['T_in'];   tout = cfg['T_out']  # しきい値を抽出
    else:
        win  = list(map(float, args.win.split(',')))  # コマンドライン入力を float 配列に変換
        wout = list(map(float, args.wout.split(',')))  # 同上 (外)
        tin  = args.tin;     tout = args.tout  # 指定しきい値を使用

    rospy.init_node('bottle_identifier')  # ROS ノードを初期化
    BottleIdentifier(
        win, wout, tin, tout,
        args.alpha, args.hysteresis_m,
        n_fields=args.n_fields
    )  # BottleIdentifier インスタンスを作成して購読を開始
    rospy.loginfo("Waiting for User Signal ...")  # 起動メッセージを出力
    rospy.spin()  # コールバック処理を継続
