import pandas as pd
import base64, os, hashlib
from Crypto.Cipher import AES

# ---------- (optional) AAD helper ----------
def _schema_hash(df: pd.DataFrame) -> bytes:
    sig = "|".join([f"{c}:{str(df[c].dtype)}" for c in df.columns]).encode("utf-8")
    return hashlib.sha256(sig).digest()

def make_aad(orig_csv_for_aad: str | None) -> bytes:
    if not orig_csv_for_aad:
        return b""
    df_schema = pd.read_csv(orig_csv_for_aad)  # full read is safest for dtypes
    return os.path.basename(orig_csv_for_aad).encode("utf-8") + b"|" + _schema_hash(df_schema)

# ---------- load key ----------
def load_key(file_path="scripts/aes_key.txt") -> bytes:
    with open(file_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    key_line = None
    for ln in lines:
        if "base64" in ln.lower() and ":" in ln:
            key_line = ln
            break
    if key_line is None:
        raise ValueError("Could not find base64 key line in aes_key.txt.")
    key_b64 = key_line.split(":", 1)[1].strip()
    key = base64.b64decode(key_b64)
    if len(key) != 32:
        raise ValueError("AES key is not 256-bit.")
    return key

# ---------- helpers ----------
def _maybe_b64(s: str) -> bytes | None:
    """Return decoded bytes if s looks like base64; else None."""
    try:
        b = base64.b64decode(s, validate=True)
        return b
    except Exception:
        return None

def decrypt_value_tolerant(value, key: bytes, aad: bytes):
    """Decrypt if cell is base64(nonce||tag||ct); else return as-is (or None for empty)."""
    if pd.isna(value) or value == "":
        return None
    if not isinstance(value, str):
        # e.g., numbers already plain
        return value
    blob = _maybe_b64(value)
    if blob is None or len(blob) < 12 + 16 + 1:
        # Not an encrypted blob; pass through
        return value
    nonce, tag, ct = blob[:12], blob[12:28], blob[28:]
    try:
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        if aad:
            cipher.update(aad)
        pt = cipher.decrypt_and_verify(ct, tag)
        return pt.decode("utf-8")
    except Exception:
        # If decryption fails (wrong key/AAD), leave value unchanged
        return value

def decrypt_dataframe_tolerant(df_enc: pd.DataFrame, key: bytes, aad: bytes) -> pd.DataFrame:
    out = df_enc.copy()
    for col in df_enc.columns:
        out[col] = df_enc[col].apply(lambda v: decrypt_value_tolerant(v, key, aad))
    return out

if __name__ == "__main__":
    # === Step 1: Load key ===
    print("[1/4] Loading key...")
    key = load_key("scripts/aes_key.txt")
    print("    ✅ Key loaded (32 bytes)")

    # Set to None if you did NOT use AAD at encryption time
    ORIG_CSV_FOR_AAD = None  # e.g., "dataset/dataset.csv"
    aad = make_aad(ORIG_CSV_FOR_AAD)

    # === Step 2: Load encrypted dataset ===
    enc_path = "scripts/encrypted_dataset.csv"
    print("[2/4] Loading encrypted dataset...")
    df_encrypted = pd.read_csv(enc_path)
    print(f"    ✅ Encrypted dataset: {df_encrypted.shape[0]} rows x {df_encrypted.shape[1]} cols")

    # === Step 3: Decrypt (tolerant) ===
    print("[3/4] Decrypting (tolerant: decrypt blobs, pass-through others)...")
    df_decrypted = decrypt_dataframe_tolerant(df_encrypted, key, aad)
    print("    ✅ Decryption pass complete")

    # Try to cast numerics back
    for col in df_decrypted.columns:
        try:
            df_decrypted[col] = pd.to_numeric(df_decrypted[col])
        except Exception:
            pass

    # === Step 4: Save decrypted CSV ===
    out_path = "scripts/decrypted_dataset.csv"
    df_decrypted.to_csv(out_path, index=False, encoding="utf-8")
    print(f"    💾 Saved: {out_path}")

    print("\nDecrypted DataFrame (first 5 rows):")
    print(df_decrypted.head())
