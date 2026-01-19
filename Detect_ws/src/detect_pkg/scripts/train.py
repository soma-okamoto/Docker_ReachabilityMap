#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline       import make_pipeline
from sklearn.preprocessing  import StandardScaler
from sklearn.linear_model   import LogisticRegressionCV
import sklearn_crfsuite
from sklearn_crfsuite       import metrics

def train_baseline(csv_path, model_out):
    df = pd.read_csv(csv_path)
    touch = df[df.touch_flag == 1]
    X = touch[['s_hand','s_head','s_accel','s_gaze']]
    y = touch['ID'].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=5, cv=3, multi_class='multinomial',
            solver='lbfgs', max_iter=300, n_jobs=-1, random_state=1
        )
    )
    pipe.fit(X_tr, y_tr)
    print(metrics.classification_report(y_te, pipe.predict(X_te), zero_division=0))
    print(metrics.confusion_matrix(y_te, pipe.predict(X_te)))
    dump(pipe, model_out)
    print(f"Baseline model saved to {model_out}")

def train_crf(csv_path, model_out):
    df = pd.read_csv(csv_path)
    X_seqs, y_seqs = [], []
    for seq_id, grp in df.groupby('sequence_id', sort=False):
        touched = grp[grp.touch_flag == 1]
        if touched.empty: continue
        tgt = int(touched.iloc[0].ID)
        seq = grp[grp.ID == tgt].sort_values('timestamp')
        X_seqs.append([
            {'s_hand':r.s_hand, 's_head':r.s_head,
             's_accel':r.s_accel, 's_gaze':r.s_gaze}
            for _,r in seq.iterrows()
        ])
        y_seqs.append([str(int(r.touch_flag)) for _,r in seq.iterrows()])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_seqs, y_seqs, test_size=0.2, random_state=0
    )
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs', c1=0.1, c2=0.1,
        max_iterations=100, all_possible_transitions=True
    )
    crf.fit(X_tr, y_tr)
    y_pred = crf.predict(X_te)
    print("CRF F1:", metrics.flat_f1_score(y_te, y_pred, average='weighted', labels=['0','1']))
    print(metrics.flat_classification_report(y_te, y_pred, labels=['0','1']))
    dump(crf, model_out)
    print(f"CRF model saved to {model_out}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task',     choices=['baseline','crf'], required=True)
    p.add_argument('--csv',      type=str, required=True, help='input CSV path')
    p.add_argument('--out',      type=str, required=True, help='output model path')
    args = p.parse_args()

    # python train.py --task baseline --csv bottle_logs.csv --out bottle_classifier.joblib
    # python train.py --task crf --csv bottle_logs.csv --out bottle_crf_model.joblib

    if args.task == 'baseline':
        train_baseline(args.csv, args.out)
    else:
        train_crf(args.csv, args.out)
