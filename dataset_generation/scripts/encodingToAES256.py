#!/usr/bin/env python3
# file: encrypt_cells_aesgcm.py
# Usage: python encrypt_cells_aesgcm.py --csv ../dataset/uciWomenBreastCancerData.csv

import argparse, os, base64, csv
from pathlib import Path
import pandas as pd
from Crypto.Cipher import AES

def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")

def aesgcm_encrypt(value_str: str, key: bytes):
    """
    Returns (blob_b64, nonce_b64, tag_b64) where:
      blob_b64 = base64(nonce || tag || ciphertext)
    """
    nonce = os.urandom(12)  # 96-bit nonce (GCM standard)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ct, tag = cipher.encrypt_and_digest(value_str.encode("utf-8"))
    blob = nonce + tag + ct
    return b64e(blob), b64e(nonce), b64e(tag)

def main():
    ap = argparse.ArgumentParser(description="Encrypt CSV cell-by-cell with AES-256-GCM")
    ap.add_argument("--csv", required=True, help="Path to plaintext CSV")
    ap.add_argument("--out", default="encrypted_dataset.csv", help="Output encrypted CSV path")
    ap.add_argument("--key_out", default="aes_key.txt", help="Where to save the base64 AES key")
    ap.add_argument("--nonces_out", default="nonces_cells.csv", help="Sidecar CSV of nonces/tags per cell")
    args = ap.parse_args()

    src = Path(args.csv)
    assert src.exists(), f"CSV not found: {src}"

    print("[1/4] Loading dataset...")
    df = pd.read_csv(src)
    df = df.drop(columns=["Unnamed: 32"], errors="ignore")
    print(f"    ✅ Loaded {df.shape[0]} rows x {df.shape[1]} cols")

    print("[2/4] Generating AES-256 key...")
    key = os.urandom(32)  # 256-bit key
    print("    ✅ Key generated")

    print("[3/4] Encrypting dataset cell-by-cell (GCM, unique nonce per cell)...")
    # Sidecar for nonces/tags
    with open(args.nonces_out, "w", newline="") as sc:
        writer = csv.writer(sc)
        writer.writerow(["row_index", "column_name", "nonce_b64", "tag_b64"])

        def enc_cell(val, r, col):
            s = "" if pd.isna(val) else str(val)
            blob_b64, nonce_b64, tag_b64 = aesgcm_encrypt(s, key)
            writer.writerow([r, col, nonce_b64, tag_b64])
            return blob_b64

        # Build encrypted DataFrame
        enc_df = df.copy()
        # Fast iteration without dtype surprises
        for r_idx in range(len(df)):
            row = df.iloc[r_idx]
            for col in df.columns:
                enc_df.iat[r_idx, df.columns.get_loc(col)] = enc_cell(row[col], r_idx, col)

    enc_df.to_csv(args.out, index=False)
    print(f"    ✅ Encrypted CSV -> {args.out}")
    print(f"    ✅ Nonces/tags   -> {args.nonces_out}")

    print("[4/4] Saving AES key (base64)...")
    with open(args.key_out, "w") as f:
        f.write("AES-256 Key (base64): " + b64e(key) + "\n")
        f.write("Notes: Each cell has a unique 12-byte nonce; see nonces_cells.csv for nonce/tag per cell.\n")
    print(f"    ✅ Key saved -> {args.key_out}")

    print("Done.")

if __name__ == "__main__":
    main()
