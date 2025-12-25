#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline       import make_pipeline
from sklearn.preprocessing  import StandardScaler
from sklearn.linear_model   import LogisticRegressionCV
from sklearn.metrics        import classification_report, confusion_matrix
from joblib                 import dump

# 1. ログCSVを読み込む
df = pd.read_csv('bottle_logs.csv')

# 2. 掴んだ瞬間のサンプルだけを取り出す
#    touch_flag==1 の行が“掴んだ”フレームに対応
touch_df = df[df['touch_flag'] == 1]

# 3. 特徴量とラベルを準備
#    s_hand, s_head, s_accel, s_gaze の4つをモデルの入力とし、
#    ID列を正解ラベルに使う
X = touch_df[['s_hand','s_head','s_accel','s_gaze']]
y = touch_df['ID'].astype(int)

# 4. 学習用とテスト用に分割
#    stratify=y として、各IDの出現比率を保ったまま分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=1,
    stratify=y
)

# 5. パイプラインで標準化→CV付きロジスティック回帰
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegressionCV(
        Cs=5,               # 正則化強度を5段階で探索
        cv=3,               # 3分割クロスバリデーション
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=300,
        n_jobs=-1,
        random_state=1
    )
)

# 6. モデルを学習
pipeline.fit(X_train, y_train)

# 7. テストセットで性能を評価
y_pred = pipeline.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, zero_division=0))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# 8. 学習済みモデルをファイルに保存
dump(pipeline, 'bottle_classifier.joblib')
print("Saved trained model to bottle_classifier.joblib")
