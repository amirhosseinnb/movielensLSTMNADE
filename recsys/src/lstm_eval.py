import os, argparse, json
import numpy as np
import pandas as pd
import torch
from torch import nn

# --- Metrics ---
def hr_at_k(rank, gt, k): return 1.0 if gt in rank[:k] else 0.0
def ndcg_at_k(rank, gt, k):
    if gt in rank[:k]:
        i = rank.index(gt)
        return 1.0 / np.log2(i + 2)
    return 0.0
def prec_at_k(rank, gt, k): return (1.0 / k) if gt in rank[:k] else 0.0
def rec_at_k(rank, gt, k):  return 1.0 if gt in rank[:k] else 0.0
def mrr_at_k(rank, gt, k):
    if gt in rank[:k]:
        i = rank.index(gt)
        return 1.0 / (i + 1)
    return 0.0

# --- Model (same as train) ---
class LSTMRec(nn.Module):
    def __init__(self, num_items, emb=64, hid=64):
        super().__init__()
        self.emb  = nn.Embedding(num_items+1, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.out  = nn.Linear(hid, num_items+1)
    def forward(self, seq):
        x = self.emb(seq)
        _, (h, _) = self.lstm(x)
        return self.out(h[-1])  # (B, num_items+1)

def main(args):
    device = torch.device("cpu")  # ارزیابی روی CPU
    # Load data
    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val   = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    test  = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    num_items = int(max(train["movieId"].max(), val["movieId"].max(), test["movieId"].max()))

    # Load model
    ckpt = torch.load(os.path.join(args.model_dir, "lstm.pt"), map_location=device)
    model = LSTMRec(num_items=num_items, emb=ckpt.get("emb",64), hid=ckpt.get("hid",64))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Build user histories from train
    user_hist = train.groupby("userId")["movieId"].apply(list).to_dict()
    val_last  = val.groupby("userId")["movieId"].first().to_dict()

    all_items = set(pd.concat([train["movieId"], val["movieId"], test["movieId"]]).unique())
    rng = np.random.default_rng(42)

    def rank_for_user(u, gt):
        hist = user_hist.get(u, [])
        if not hist: 
            return None
        # candidate set: 1 positive + N negatives that user hasn't seen
        seen = set(hist)
        if u in val_last: seen.add(int(val_last[u]))
        seen.add(gt)
        pool = list(all_items - seen)
        N = min(args.neg, len(pool))
        negs = list(rng.choice(pool, size=N, replace=False)) if N > 0 else []
        cand = [gt] + negs

        x = torch.tensor(hist, dtype=torch.long).unsqueeze(0)  # (1, T)
        with torch.no_grad():
            logits = model(x).squeeze(0)  # (num_items+1,)
        scores = [(i, float(logits[i].item())) for i in cand]
        scores.sort(key=lambda t: t[1], reverse=True)
        return [i for i, _ in scores]

    Ks = [10, 20]
    buckets = {f"HR@{k}": [] for k in Ks}
    buckets.update({f"NDCG@{k}": [] for k in Ks})
    buckets.update({f"Precision@{k}": [] for k in Ks})
    buckets.update({f"Recall@{k}": [] for k in Ks})
    buckets.update({f"MRR@{k}": [] for k in Ks})

    for row in test.itertuples(index=False):
        u, gt = int(row.userId), int(row.movieId)
        rank = rank_for_user(u, gt)
        if rank is None: 
            continue
        for K in Ks:
            buckets[f"HR@{K}"].append(hr_at_k(rank, gt, K))
            buckets[f"NDCG@{K}"].append(ndcg_at_k(rank, gt, K))
            buckets[f"Precision@{K}"].append(prec_at_k(rank, gt, K))
            buckets[f"Recall@{K}"].append(rec_at_k(rank, gt, K))
            buckets[f"MRR@{K}"].append(mrr_at_k(rank, gt, K))

    results = {m: float(np.mean(v)) for m, v in buckets.items()}
    os.makedirs(args.out_dir, exist_ok=True)
    outp = os.path.join(args.out_dir, "lstm_results.json")
    with open(outp, "w") as f: json.dump(results, f, indent=2, ensure_ascii=False)
    print("saved:", outp)
    print(results)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  type=str, default="recsys/data")
    p.add_argument("--model_dir", type=str, default="recsys/models")
    p.add_argument("--out_dir",   type=str, default="recsys/results")
    p.add_argument("--neg",       type=int, default=100)  # تعداد منفی‌های نمونه‌گیری‌شده
    args = p.parse_args()
    main(args)
