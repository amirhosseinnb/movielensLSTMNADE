import os, zipfile, argparse
import pandas as pd

def resolve_u_data(src_path: str, data_raw: str) -> str:
    """
    Accepts either:
      - a directory that contains 'u.data' or 'ml-100k/u.data'
      - a zip file (ml-100k.zip), which will be extracted to data_raw/
    Returns absolute path to u.data
    """
    src_path = os.path.abspath(src_path)
    data_raw = os.path.abspath(data_raw)

    if os.path.isdir(src_path):
        cand1 = os.path.join(src_path, "u.data")
        cand2 = os.path.join(src_path, "ml-100k", "u.data")
        if os.path.exists(cand1): return cand1
        if os.path.exists(cand2): return cand2
        raise FileNotFoundError("u.data not found inside the provided directory.")
    else:
        # assume it's a zip
        if not zipfile.is_zipfile(src_path):
            raise ValueError("Provided --src is neither a directory nor a valid zip file.")
        os.makedirs(data_raw, exist_ok=True)
        with zipfile.ZipFile(src_path, 'r') as z:
            z.extractall(data_raw)
        # typical structure after extract: data_raw/ml-100k/u.data
        cand = os.path.join(data_raw, "ml-100k", "u.data")
        if os.path.exists(cand): return cand
        # sometimes an extra folder level
        cand2 = os.path.join(data_raw, "ml-100k", "ml-100k", "u.data")
        if os.path.exists(cand2): return cand2
        raise FileNotFoundError("u.data not found after extracting the zip.")

def main(args):
    u_data_path = resolve_u_data(args.src, args.data_raw)

    cols = ["userId","movieId","rating","timestamp"]
    df = pd.read_csv(u_data_path, sep="\t", names=cols, engine="python").astype(
        {"userId":int,"movieId":int,"rating":float,"timestamp":int}
    )

    # implicit positive
    df = df[df["rating"] >= args.pos_threshold].copy()
    # sort by time
    df = df.sort_values(["userId","timestamp"])
    # min positives per user
    counts = df.groupby("userId")["movieId"].count()
    valid_users = counts[counts >= args.min_pos].index
    df = df[df["userId"].isin(valid_users)].copy()

    # temporal LOO per user
    trains, vals, tests = [], [], []
    for uid, g in df.groupby("userId", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 3:
            continue
        test = g.iloc[[-1]]
        val  = g.iloc[[-2]]
        train= g.iloc[:-2]
        trains.append(train); vals.append(val); tests.append(test)

    train_df = pd.concat(trains, ignore_index=True)
    val_df   = pd.concat(vals,   ignore_index=True)
    test_df  = pd.concat(tests,  ignore_index=True)

    os.makedirs(args.out_dir, exist_ok=True)
    train_df[["userId","movieId","timestamp"]].to_csv(os.path.join(args.out_dir,"train.csv"), index=False)
    val_df[["userId","movieId","timestamp"]].to_csv(os.path.join(args.out_dir,"val.csv"), index=False)
    test_df[["userId","movieId","timestamp"]].to_csv(os.path.join(args.out_dir,"test.csv"), index=False)

    print({
        "users": df["userId"].nunique(),
        "items": df["movieId"].nunique(),
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df)
    })

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="path to ml-100k folder OR ml-100k.zip")
    p.add_argument("--data_raw", default="recsys/data_raw", help="where to extract zip (if src is a zip)")
    p.add_argument("--out_dir",  default="recsys/data",     help="where to write train/val/test CSVs")
    p.add_argument("--pos_threshold", type=float, default=4.0)
    p.add_argument("--min_pos", type=int, default=5)
    args = p.parse_args()
    main(args)
