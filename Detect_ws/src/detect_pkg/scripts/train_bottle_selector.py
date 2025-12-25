# train_bottle_selector.py
#!/usr/bin/env python3
import pandas as pd
import numpy as np
torch_imported = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    torch_imported = True
except ImportError:
    print("PyTorch is required. Please install with `pip install torch`.")
    exit(1)
import argparse

class BottleSequenceDataset(Dataset):
    """
    Dataset for per-bottle LSTM training. For each sequence (user reaching a bottle),
    creates samples of length seq_len for each bottle in that sequence.
    Label=1 for the actually grasped bottle, else 0.
    """
    def __init__(self, csv_path, seq_len, delta):
        df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.delta   = delta
        self.samples = []  # list of (features:seq_len×F, label:0/1)

        # group by sequence_id
        for seq_id, g in df.groupby('sequence_id'):
            g = g.sort_values('timestamp').reset_index(drop=True)
            # find grasp frames
            grasp_idxs = g.index[g['touch_flag'] == 1].tolist()
            if not grasp_idxs:
                continue
            grasp_idx = grasp_idxs[-1]
            start = grasp_idx - seq_len + 1 - delta
            end   = grasp_idx - delta
            if start < 0 or end >= len(g):
                continue
            window = g.iloc[start:end+1]
            target_id = int(g.iloc[grasp_idx]['ID'])
            # features dimension: x,y,z,reach_flag,s_hand,s_head,s_accel
            for b_id in window['ID'].unique():
                df_id = window[window['ID'] == b_id]
                if len(df_id) != seq_len:
                    continue
                feats = df_id[['x','y','z','reach_flag','s_hand','s_head','s_accel']].values
                label = 1.0 if b_id == target_id else 0.0
                self.samples.append((feats, label))

        # convert to tensors
        self.data = [ (torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
                      for x,y in self.samples ]

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class LSTMBottleSelector(nn.Module):
    """
    LSTM-based binary classifier for per-bottle sequence.
    Input: (batch, seq_len, feature_dim)
    Output: (batch,) logit for probability of grasp
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, seq_len, F)
        _, (hn, _) = self.lstm(x)
        h = hn[-1]        # (B, hidden_dim)
        logit = self.fc(h).squeeze(-1)  # (B,)
        return logit


def train(args):
    dataset = BottleSequenceDataset(args.csv, args.seq_len, args.delta)
    # データ数が0なら学習を中断
    if len(dataset) == 0:
        print(f"Error: No training samples found. seq_len={args.seq_len}, delta={args.delta}")
        print("  → CSVの'touch_flag'やシーケンス長を確認してください。")
        return
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = LSTMBottleSelector(input_dim=7, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{args.epochs}  Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"Model saved to {args.out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train per-bottle grasp predictor')
    parser.add_argument('--csv',       type=str, default='/home/dars/Detect_ws/src/detect_pkg/csv/bottle_logs.csv', help='Path to CSV log file')
    parser.add_argument('--seq_len',   type=int, default=10, help='Sequence length')
    parser.add_argument('--delta',     type=int, default=0, help='Frame offset for label')
    parser.add_argument('--hidden_dim',type=int, default=64, help='LSTM hidden size')
    parser.add_argument('--batch_size',type=int, default=32)
    parser.add_argument('--epochs',    type=int, default=20)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--out',       type=str, default='bottle_selector.pth', help='Output model path')
    args = parser.parse_args()
    train(args)