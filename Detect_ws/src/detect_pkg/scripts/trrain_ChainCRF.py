#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from joblib import dump
from sklearn.utils import resample
import numpy as np


# 1) CSV 読み込み
#    ヘッダに: timestamp,sequence_id,ID,x,y,z,reach_flag,touch_flag,s_hand,s_head,s_accel,s_gaze
df = pd.read_csv('/home/dars/Detect_ws/src/detect_pkg/csv/bottle_logs.csv')

# 例：各ターゲットシーケンス m 本ずつに揃える
seq_labels = df[df.touch_flag==1].groupby('sequence_id')['ID'].first().reset_index()
m = seq_labels['ID'].value_counts().max()
upsamp = []
for tgt, grp in seq_labels.groupby('ID'):
    upsamp.append(
      resample(grp, replace=True, n_samples=m, random_state=0)
    )
balanced = pd.concat(upsamp)
df = df[df.sequence_id.isin(balanced.sequence_id)]

delta = 3
df['touch_future'] = (
    df.groupby('sequence_id')['touch_flag']
      .shift(-delta)
      .fillna(0)
      .astype(int)
)

# 2) シーケンスごとに「掴む対象ボトル」のデータだけ抜き出す
X_seqs, y_seqs = [], []
for seq_id, grp in df.groupby('sequence_id', sort=False):
    touched = grp[grp.touch_flag == 1]
    if touched.empty:
        continue
    target = int(touched.iloc[0].ID)

    # ① 時刻ごとにまとめ直す
    timestamps = sorted(grp.timestamp.unique())
    X_seq = []
    for t in timestamps:
        feat = {}
        sub = grp[grp.timestamp == t]
        for r in sub.itertuples():
            flag = r.reach_flag
            feat[f's_hand_in_{r.ID}']   = r.s_hand  if flag==1 else 0.0
            feat[f's_hand_out_{r.ID}']  = r.s_hand  if flag==0 else 0.0
            feat[f's_head_in_{r.ID}']   = r.s_head  if flag==1 else 0.0
            feat[f's_head_out_{r.ID}']  = r.s_head  if flag==0 else 0.0
            feat[f's_accel_in_{r.ID}']  = r.s_accel if flag==1 else 0.0
            feat[f's_accel_out_{r.ID}'] = r.s_accel if flag==0 else 0.0
            # feat[f's_gaze_in_{r.ID}']  = r.s_gaze if flag==1 else 0.0
            # feat[f's_gaze_out_{r.ID}'] = r.s_gaze if flag==0 else 0.0

        X_seq.append(feat)

    # ② 同じ timestamps リストを使ってラベルを１対１で作る
    y_seq = []
    for t in timestamps:
        f = grp[grp.timestamp == t]['touch_future'].iloc[0]
        # δ フレーム先の掴み予測が立っていればターゲットID、
        # そうでなければ “0”（none）
        y_seq.append(str(target) if f == 1 else '0')

    # ここで必ず len(X_seq) == len(y_seq)
    X_seqs.append(X_seq)
    y_seqs.append(y_seq)

# 3) シーケンス単位で train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_seqs, y_seqs, test_size=0.2, random_state=0
)

# 4) CRF モデル定義
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,              # L1 正則化強度
    c2=0.1,              # L2 正則化強度
    max_iterations=100,
    all_possible_transitions=True
)

# 5) 学習
crf.fit(X_train, y_train)

# 6) 特徴重みの可視化
print("=== State feature weights ===")
for (lbl, feat), w in crf.state_features_.items():
    print(f"  ({lbl}, {feat}) = {w:.3f}")

print("\n=== Transition weights ===")
for (p, n), w in crf.transition_features_.items():
    print(f"  ({p} → {n}) = {w:.3f}")

# 7) 評価
y_pred = crf.predict(X_test)
# 全クラス（ボトルID）を検出して評価
labels = sorted(crf.classes_, key=lambda x: int(x))
print(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels))
print(metrics.flat_classification_report(y_test, y_pred, labels=labels))


# まずシーケンス単位で予測シミュレーション
false_alarms = 0
neg_seqs     = 0

for X_seq, y_true_seq in zip(X_test, y_test):
    # 正解に positive (ID) が一度もないシーケンス
    if all(f == '0' for f in y_true_seq):
        neg_seqs += 1
        y_pred_seq = crf.predict([X_seq])[0]
        # もしどこかで非ゼロを予測したら誤報
        if any(f != '0' for f in y_pred_seq):
            false_alarms += 1

# 誤報率 = 誤報した負例シーケンスの数 / 全負例シーケンス数
if neg_seqs > 0:
    fa_rate = false_alarms / neg_seqs
else:
    fa_rate = np.nan

print(f"False‑Alarm rate: {fa_rate:.3f}")




# 8) モデルの保存
dump(crf, '../Train_joblib/bottle_crf_model.joblib')
print("\nSaved CRF model to bottle_crf_model.joblib")
