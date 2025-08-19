import os, argparse, random, json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

class SeqDataset(Dataset):
    def __init__(self, df, min_len=5):
        self.user2seq = {}
        for u, g in df.sort_values(["userId","timestamp"]).groupby("userId"):
            seq = g["movieId"].tolist()
            if len(seq) >= min_len:
                self.user2seq[int(u)] = seq
        self.users = list(self.user2seq.keys())

    def __len__(self): return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        seq = self.user2seq[u]
        x = torch.tensor(seq[:-1], dtype=torch.long)  # input sequence
        y = torch.tensor(seq[-1],  dtype=torch.long)  # next item (target)
        return x, y

def collate_fn(batch):
    seqs = [b[0] for b in batch]
    ys   = torch.stack([b[1] for b in batch])
    seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)  # left-pad with 0
    return seqs, ys

class LSTMRec(nn.Module):
    def __init__(self, num_items, emb=64, hid=64):
        super().__init__()
        self.emb  = nn.Embedding(num_items+1, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.out  = nn.Linear(hid, num_items+1)

    def forward(self, seq):
        x = self.emb(seq)                # (B, T, E)
        _, (h, _) = self.lstm(x)         # h: (1, B, H)
        logits = self.out(h[-1])         # (B, num_items+1)
        return logits

def main(args):
    set_seed(42)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val   = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    test  = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    num_items = int(max(train["movieId"].max(), val["movieId"].max(), test["movieId"].max()))

    ds = SeqDataset(train, min_len=args.min_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

    model = LSTMRec(num_items=num_items, emb=args.emb, hid=args.hid).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for ep in range(1, args.epochs+1):
        total = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item())
        print(f"epoch {ep}: loss={total/len(dl):.4f}")

    os.makedirs(args.model_dir, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "num_items": num_items,
                "emb": args.emb, "hid": args.hid}, 
               os.path.join(args.model_dir, "lstm.pt"))
    print("saved:", os.path.join(args.model_dir, "lstm.pt"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  type=str, default="recsys/data")
    p.add_argument("--model_dir", type=str, default="recsys/models")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--min_len", type=int, default=5)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--emb", type=int, default=64)
    p.add_argument("--hid", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=5)
    args = p.parse_args()
    main(args)
