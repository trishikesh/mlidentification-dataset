# server_he_infer_scalars.py
import argparse, json, numpy as np, tenseal as ts, pickle
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params",   default="models/model_params.json")
    ap.add_argument("--ctx_pub",  default="he_out/ctx_pub.bin")
    ap.add_argument("--enc_list", default="he_out/enc_x_scalars.pkl")
    ap.add_argument("--out",      default="he_out/enc_mul_scalars.pkl")
    args = ap.parse_args()

    mp = json.load(open(args.params,"r"))
    w  = np.array(mp["weights"], dtype=float)

    ctx = ts.context_from(open(args.ctx_pub,"rb").read())

    # Load list of serialized encrypted scalars
    enc_scalars = pickle.load(open(args.enc_list,"rb"))
    if len(enc_scalars) != len(w):
        raise ValueError(f"Length mismatch: got {len(enc_scalars)} scalars, {len(w)} weights")

    # Multiply each scalar by its weight
    enc_mul_list = []
    for i, ser in enumerate(enc_scalars):
        v = ts.ckks_vector_from(ctx, ser)   # 1-slot vector
        enc_mul = v * w[i]                  # multiply by plaintext weight
        enc_mul_list.append(enc_mul.serialize())

    Path(args.out).parent.mkdir(exist_ok=True)
    with open(args.out,"wb") as f:
        pickle.dump(enc_mul_list, f)
    print(f"✅ Wrote encrypted scalar contributions -> {args.out}")

if __name__ == "__main__":
    main()
