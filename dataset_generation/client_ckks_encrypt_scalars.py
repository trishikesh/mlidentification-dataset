# client_ckks_encrypt_scalars.py
import argparse, json, numpy as np, pandas as pd, tenseal as ts, pickle
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="scripts/decrypted_dataset.csv")
    ap.add_argument("--params", default="models/model_params.json")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--drop", nargs="*", default=["id","Unnamed: 32","diagnosis"])
    args = ap.parse_args()

    mp   = json.load(open(args.params,"r"))
    feat = mp["feature_names"]
    mu   = np.array(mp["scaler_mean"], dtype=float)
    sig  = np.array(mp["scaler_scale"], dtype=float)
    w    = np.array(mp["weights"], dtype=float)
    b    = float(mp["bias"])

    df = pd.read_csv(args.csv)
    for c in args.drop:
        if c in df.columns: df = df.drop(columns=[c])

    x = df.loc[args.row, feat].astype(float).values
    sig_safe = np.where(sig == 0, 1e-12, sig)
    x_std = (x - mu) / sig_safe

    # Print expected plaintext score for sanity
    score_plain = float(np.dot(x_std, w) + b)
    print(f"[client] Plain expected score: {score_plain:.6f}")

    # CKKS context (client keeps SK)
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[40,21,21,40]
    )
    ctx.global_scale = 2**40
    # No rotations needed, so Galois keys not required

    # Encrypt each feature as a 1-slot vector
    enc_scalars = []
    for xi in x_std.tolist():
        enc_scalars.append(ts.ckks_vector(ctx, [xi]).serialize())

    Path("he_out").mkdir(exist_ok=True)
    # Save list of serialized scalars in one pickle
    with open("he_out/enc_x_scalars.pkl","wb") as f:
        pickle.dump(enc_scalars, f)

    # Public and secret contexts
    open("he_out/ctx_pub.bin","wb").write(ctx.serialize(save_secret_key=False))
    open("he_out/ctx_with_sk.bin","wb").write(ctx.serialize(save_secret_key=True))

    print("✅ Wrote he_out/enc_x_scalars.pkl, ctx_pub.bin (to server), ctx_with_sk.bin (KEEP SECRET)")

if __name__ == "__main__":
    main()
