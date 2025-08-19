import os, zipfile, argparse
import pandas as pd

def ensure_extract(zip_path, out_base):
    raw_dir = os.path.join(out_base, "ml-100k")
    if not os.path.exists(raw_dir):
        os.makedirs(out_base, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(out_base)
    return raw_dir

def main(args):
    raw_dir = ensure_extract(args.zip_path, args.data_raw)
    u_data_path = os.path.join(raw_dir, "ml-100k", "u.data")
    cols = ["userId","movieId","rating","timestamp"]
    df = pd.read_csv(u_data_path, sep="\t", names=cols, engine="python").astype(
        {"userId":int,"movieId":int,"rating":float,"timestamp":int}
    )
    df = df[df["rating"] >= args.pos_threshold].sort_values(["userId","timestamp"])
    valid = df.groupby("userId")["movieId"].count()
    df = df[df["userId"].isin(valid[valid >= args.min_pos].index)]

    trains, vals, tests = [], [], []
    for uid, g in df.groupby("userId", sort=False):
        g = g.reset_index(drop=True)
        test = g.iloc[[-1]]; val = g.iloc[[-2]]; train = g.iloc[:-2]
        trains.append(train); vals.append(val); tests.append(test)

    train_df = pd.concat(trains, ignore_index=True)
    val_df   = pd.concat(vals,   ignore_index=True)
    test_df  = pd.concat(tests,  ignore_index=True)

    os.makedirs(args.out_dir, exist_ok=True)
    train_df[["userId","movieId","timestamp"]].to_csv(os.path.join(args.out_dir,"train.csv"), index=False)
    val_df[["userId","movieId","timestamp"]].to_csv(os.path.join(args.out_dir,"val.csv"), index=False)
    test_df[["userId","movieId","timestamp"]].to_csv(os.path.join(args.out_dir,"test.csv"), index=False)

    print({"users": df["userId"].nunique(), "items": df["movieId"].nunique(),
           "train": len(train_df), "val": len(val_df), "test": len(test_df)})

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--zip_path", required=True)
    p.add_argument("--data_raw", default="recsys/data_raw")
    p.add_argument("--out_dir",  default="recsys/data")
    p.add_argument("--pos_threshold", type=float, default=4.0)
    p.add_argument("--min_pos", type=int, default=5)
    args = p.parse_args()
    main(args)
