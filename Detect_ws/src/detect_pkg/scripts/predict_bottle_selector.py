# predict_bottle_selector.py
#!/usr/bin/env python3
import pandas as pd
import torch
import argparse
from train_bottle_selector import LSTMBottleSelector


def predict(args):
    # load model
    model = LSTMBottleSelector(input_dim=7, hidden_dim=args.hidden_dim)
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()

    # read CSV and select sequence
    df = pd.read_csv(args.csv)
    seq = df[df['sequence_id'] == args.seq_id].sort_values('timestamp').reset_index(drop=True)
    idxs = seq.index[seq['touch_flag'] == 1].tolist()
    if not idxs:
        print('No grasp event in sequence', args.seq_id)
        return
    grasp_idx = idxs[-1]
    start = grasp_idx - args.seq_len + 1 - args.delta
    end   = grasp_idx - args.delta
    if start < 0 or end >= len(seq):
        print('Sequence too short for seq_len/delta')
        return
    window = seq.iloc[start:end+1]

    # per-bottle scoring
    scores = []  # (bottle_id, logit)
    for b_id in window['ID'].unique():
        df_id = window[window['ID'] == b_id]
        if len(df_id) != args.seq_len:
            continue
        feats = df_id[['x','y','z','reach_flag','s_hand','s_head','s_accel']].values
        x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logit = model(x).item()
        scores.append((b_id, logit))
    if not scores:
        print('No valid bottle windows found')
        return
    # pick highest
    pred = max(scores, key=lambda t: t[1])[0]
    print('Predicted bottle to grasp:', pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict future grasped bottle')
    parser.add_argument('--model',    type=str, required=True)
    parser.add_argument('--csv',      type=str, required=True)
    parser.add_argument('--seq_id',   type=int, required=True)
    parser.add_argument('--seq_len',  type=int, default=10)
    parser.add_argument('--delta',    type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    args = parser.parse_args()
    predict(args)
