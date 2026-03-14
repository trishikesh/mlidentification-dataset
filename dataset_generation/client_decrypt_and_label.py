# client_decrypt_and_label.py  (auto-calibrates scale; verbose debug)
import argparse, json, math, numpy as np, tenseal as ts, pickle
import pandas as pd
from pathlib import Path

def stable_sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z); return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z);  return ez / (1.0 + ez)

def load_params(p):
    mp = json.load(open(p, "r"))
    return (
        mp["feature_names"],
        np.array(mp["weights"], dtype=float),
        float(mp["bias"]),
        np.array(mp["scaler_mean"], dtype=float),
        np.array(mp["scaler_scale"], dtype=float),
    )

def expected_contribs(csv, row, feat, w, mu, sigma, drop):
    df = pd.read_csv(csv)
    for c in drop:
        if c in df.columns: df = df.drop(columns=[c])
    x = df.loc[row, feat].astype(float).values
    sig_safe = np.where(sigma == 0, 1e-12, sigma)
    x_std = (x - mu) / sig_safe
    return x_std * w, float(np.dot(x_std, w))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx_with_sk", default="he_out/ctx_with_sk.bin")
    ap.add_argument("--enc_list",    default="he_out/enc_mul_scalars.pkl")
    ap.add_argument("--params",      default="models/model_params.json")
    ap.add_argument("--csv",         default="scripts/decrypted_dataset.csv", help="Plaintext CSV used for calibration")
    ap.add_argument("--row",         type=int, default=0)
    ap.add_argument("--drop",        nargs="*", default=["id","Unnamed: 32","diagnosis"])
    ap.add_argument("--threshold",   type=float, default=0.5)
    ap.add_argument("--debug",       action="store_true")
    args = ap.parse_args()

    feat, w, b, mu, sigma = load_params(args.params)
    n_features = len(w)

    # 1) Decrypt scalar contributions
    ctx = ts.context_from(open(args.ctx_with_sk,"rb").read())
    enc_mul_list = pickle.load(open(args.enc_list,"rb"))
    if len(enc_mul_list) != n_features:
        raise ValueError(f"Got {len(enc_mul_list)} scalars, expected {n_features}")

    dec = []
    for ser in enc_mul_list:
        v = ts.ckks_vector_from(ctx, ser)   # 1-slot vector
        dec.append(v.decrypt()[0])
    dec = np.array(dec, dtype=float)

    # 2) Compute expected plaintext contribs for same row
    exp_contribs, exp_sum = expected_contribs(args.csv, args.row, feat, w, mu, sigma, args.drop)

    # 3) Estimate scale = median(|dec/exp|) on non-zero expected entries
    mask = np.abs(exp_contribs) > 1e-9
    if not np.any(mask):
        raise RuntimeError("All expected contributions are ~0; choose a different row.")
    ratios = np.abs(dec[mask] / exp_contribs[mask])
    scale = float(np.median(ratios))

    # 4) Apply scale to decrypted contribs
    contribs = dec / (scale if scale != 0 else 1.0)

    # 5) Compute totals
    sum_contribs = float(contribs.sum())
    score = sum_contribs + b
    prob  = stable_sigmoid(score)
    label = 1 if prob >= args.threshold else 0

    # 6) Debug prints
    print(f"[calibration] estimated scale ≈ {scale:.6f}")
    if args.debug:
        print(f"[debug] first5 dec (raw):      {dec[:5].round(6).tolist()}")
        print(f"[debug] first5 expected:       {exp_contribs[:5].round(6).tolist()}")
        print(f"[debug] first5 after scaling:  {contribs[:5].round(6).tolist()}")
    print(f"[pieces] sum_contribs={sum_contribs:.6f}   bias={b:.6f}")
    print(f"score={score:.6f}  prob={prob:.6f}  pred={label} (1=malignant, 0=benign)")
    print(f"[sanity] plaintext dot={exp_sum:.6f}  (score should be ≈ dot + bias)")

if __name__ == "__main__":
    main()
