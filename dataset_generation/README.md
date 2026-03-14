# Privacy-Preserving Breast Cancer Inference (CKKS + AES-GCM)

> End‑to‑end demo pipeline combining: classical ML (Logistic Regression), symmetric encryption at rest (AES‑256‑GCM, cell‑level), and _partial_ homomorphic evaluation (TenSEAL CKKS) of a linear model (encrypted feature \* weight products) with client‑side decryption, rescaling and classification.

---

## 📌 High-Level Overview

| Phase                      | Purpose                                                   | Main Files                                                 |
| -------------------------- | --------------------------------------------------------- | ---------------------------------------------------------- |
| Data Encryption (optional) | Protect raw CSV at rest with AES‑256‑GCM (per cell)       | `scripts/encodingToAES256.py`, `aes_decrypt_all_in_RAM.py` |
| Model Training & Export    | Train Logistic Regression & export minimal params         | `train_and_export_model.py`, `models/*`                    |
| Plain Debug                | Verify plaintext score for a chosen row                   | `debug_plain_score.py`                                     |
| Client Encrypt             | Standardize one row & encrypt each feature as CKKS scalar | `client_ckks_encrypt_scalars.py`                           |
| Server HE Multiply         | Multiply encrypted features by plaintext weights          | `server_he_infer_scalars.py`                               |
| Client Decrypt & Classify  | Decrypt weighted sums, rescale, add bias, sigmoid, label  | `client_decrypt_and_label.py`                              |

Only the element‑wise products are performed homomorphically. Summation of contributions and final logistic sigmoid happen in plaintext after calibrated scale recovery.

---

## 🔐 Cryptographic / ML Design

- **Model**: Binary Logistic Regression trained on UCI Breast Cancer (diagnosis M/B mapped to 1/0). Exported artifacts:
  - Full objects (`models/logreg.joblib`, `models/scaler.joblib`).
  - Minimal JSON (`models/model_params.json`): ordered `feature_names`, `weights`, `bias`, `scaler_mean`, `scaler_scale`.
- **AES‑256‑GCM (optional layer)**:
  - `encodingToAES256.py` encrypts every CSV cell with a unique 96‑bit nonce; output: `encrypted_dataset.csv` + sidecar `nonces_cells.csv` + key file `aes_key.txt` (base64).
  - `aes_decrypt_all_in_RAM.py` tolerantly decrypts values back (keeps non‑encrypted strings / numeric passthrough). Supports future schema‑bound AAD (currently disabled by default).
- **CKKS / TenSEAL**:
  - Each standardized feature value (1 real number) is packed alone in a CKKS vector of size 1 (no rotations required).
  - Server receives: serialized public context (`ctx_pub.bin`) and list of encrypted standardized scalars (`enc_x_scalars.pkl`).
  - Server multiplies each by its plaintext weight ⇒ encrypted contributions (`enc_mul_scalars.pkl`). No additions to avoid scale drift complexity here.
  - Client loads secret context (`ctx_with_sk.bin`), decrypts each contribution, estimates a global scale via median(|dec/expected|) using the plaintext row (calibration), divides to recover approximate true contributions, sums, adds bias, applies sigmoid.
  - Threshold (default 0.5) yields prediction (1 malignant / 0 benign).

This design demonstrates split trust: server never sees raw features or model secret key; client never reveals feature plaintext during weighted multiplication.

---

## 🗂 File-by-File Explanation

### Root Scripts

- **`train_and_export_model.py`**

  1. Loads and cleans dataset (drops `id`, `Unnamed: 32`, encodes target).
  2. Splits, scales, trains Logistic Regression (`max_iter=1000`).
  3. Evaluates and exports artifacts (joblib + JSON). Performs a reproducibility sanity check using only JSON values.

- **`main.py`** – Simpler training script printing progress + classification report (no export JSON verification extras).

- **`app.py`** – Alternative model: Random Forest classifier baseline for comparison (not used in HE flow).

- **`debug_plain_score.py`** – Loads JSON params & decrypted dataset row, recomputes plaintext logistic score for a given row to compare with HE pipeline output.

- **`client_ckks_encrypt_scalars.py`**

  - Loads model params JSON and a (decrypted) CSV.
  - Standardizes one selected row using stored scaler mean/scale (safe guard for zero variance: substitute tiny epsilon).
  - Serializes CKKS context (with and without secret key) and encrypts each standardized feature individually.
  - Outputs: `he_out/enc_x_scalars.pkl`, `he_out/ctx_pub.bin`, `he_out/ctx_with_sk.bin`.

- **`server_he_infer_scalars.py`**

  - Receives encrypted standardized scalars + public context + plaintext weights.
  - Performs element-wise multiplications (ciphertext \* plaintext double) for each dimension.
  - Saves encrypted contributions list `enc_mul_scalars.pkl` (returned to client).

- **`client_decrypt_and_label.py`**

  - Decrypts each encrypted contribution with secret context.
  - Recomputes expected plaintext contributions from local row to estimate CKKS scale (median ratio robust to outliers).
  - Rescales decrypted numbers, sums, adds bias, applies a stable sigmoid, thresholds.
  - Provides extensive debug outputs (`--debug`).

- **`aes_decrypt_all_in_RAM.py`**
  - Loads AES key (256-bit) from `scripts/aes_key.txt` (expects base64 marker line).
  - Tolerantly attempts to base64-decode each cell into nonce|tag|ciphertext and decrypt; silently passes through on failure.
  - Optionally supports constructing Additional Authenticated Data from original CSV name + schema hash (commented usage in script).

### `scripts/` Directory

- **`encodingToAES256.py`** (named in header as `encrypt_cells_aesgcm.py`):
  - Generates fresh AES-256 key; encrypts each cell (GCM, unique nonce) writing:
    - Encrypted dataset: `scripts/encrypted_dataset.csv` (base64 blobs per cell: nonce||tag||ct).
    - Nonce/tag mapping: `scripts/nonces_cells.csv` (auditing / optional forensic use).
    - Key file: `scripts/aes_key.txt` (base64 key + notes).
  - Drops spurious `Unnamed: 32` column.
- **Data artifacts**: `encrypted_dataset.csv`, `decrypted_dataset.csv` (produced after decryption), `nonces_cells.csv`, `aes_key.txt`.

### `models/` Directory

- `logreg.joblib` – Trained scikit-learn LogisticRegression.
- `scaler.joblib` – StandardScaler fitted on training data.
- `model_params.json` – Minimal reproducible inference bundle:

```jsonc
{
  "feature_names": [...],           // enforced feature order
  "weights": [...],                 // logistic regression coefficients
  "bias": float,                    // intercept
  "scaler_mean": [...],             // per-feature means
  "scaler_scale": [...]             // per-feature scales
}
```

### `he_out/` Directory (ephemeral HE session outputs)

- `ctx_pub.bin` – TenSEAL context without secret key (safe to send to server).
- `ctx_with_sk.bin` – Context with secret key (KEEP PRIVATE).
- `enc_x_scalars.pkl` – Pickled list of serialized CKKS vectors (each a single standardized feature value).
- `enc_mul_scalars.pkl` – Encrypted feature \* weight contributions returned by server.

---

## 🔄 Data & Computation Flow

```
                +------------------+          +------------------+
Plain CSV  ---> | (Optional) AES   |  ---->   | Encrypted CSV    |
(dataset/*.csv) | encrypt_cells... |          | + key + nonces   |
                +------------------+                 |
                        | (decrypt to work)          v
                        |                     decrypted_dataset.csv
                        v
             +-----------------------+
             | train_and_export_model|  -> model_params.json + scaler.joblib + logreg.joblib
             +-----------+-----------+
                         |
                         v (select row)
             +-----------------------------+
             | client_ckks_encrypt_scalars | -- ctx_pub.bin -->
             |  (standardize & encrypt)    | -- enc_x_scalars.pkl -->
             +---------------+-------------+                       (untrusted)
                             |                                    SERVER
                             v                                           |
                        (private)                                       v
                       ctx_with_sk.bin        +-------------------------------+
                                              | server_he_infer_scalars       |
                                              | (multiply by plaintext weights)|
                                              +----------------+--------------+
                                                               |
                                                               v
                                          enc_mul_scalars.pkl (encrypted contribs)
                                                               |
                                                               v
                      +-----------------------------------------------+
                      | client_decrypt_and_label                      |
                      |  decrypt -> rescale -> sum -> +bias -> sigmoid|
                      +-----------------------------------------------+
```

---

## 🧮 Math (Logistic Regression + HE Step)

Given standardized feature vector \( x_s = (x - \mu) / \sigma \), weights \( w \), bias \( b \):

- Plain score (logit): \( z = w^T x_s + b \)
- Probability: \( p = 1/(1+e^{-z}) \)
- Prediction: \( \hat{y} = 1[p \ge 0.5] \)

HE pipeline computes encrypted partials:
\( Enc(x*{s,i}) \* w_i = Enc(x*{s,i} w*i) \)
Returning decrypted approximations \( d_i = s \* (x*{s,i} w_i) + \epsilon_i \) where `s` is CKKS scale. Client estimates `s` via median ratio and rescales: \( \tilde{c}\_i = d_i / s \). Then \( z \approx \sum_i \tilde{c}\_i + b \).

---

## 🧪 Reproducibility & Debug

- Use `debug_plain_score.py` to verify `z` matches (within small float error) the HE reconstructed score for the same row index.
- Scaling calibration uses robust median to mitigate noise/outliers.

---

## ▶️ How to Run (Step-by-Step)

Prerequisites: Python 3.10+, install dependencies (scikit-learn, pandas, numpy, joblib, pycryptodome, tenseal).

1. (Optional) AES encrypt original dataset

```powershell
python scripts/encodingToAES256.py --csv dataset/uciWomenBreastCancerData.csv --out scripts/encrypted_dataset.csv
```

2. (Optional) Decrypt to working copy (produces `scripts/decrypted_dataset.csv`)

```powershell
python aes_decrypt_all_in_RAM.py
```

Or skip steps 1–2 and just copy the original CSV to `scripts/decrypted_dataset.csv` if you don't need at-rest encryption demo.

3. Train model & export params

```powershell
python train_and_export_model.py
```

4. Encrypt one row's standardized features (choose `--row` index)

```powershell
python client_ckks_encrypt_scalars.py --row 0
```

5. Server multiplies by plaintext weights

```powershell
python server_he_infer_scalars.py
```

6. Client decrypts, rescales, classifies (optionally `--debug`)

```powershell
python client_decrypt_and_label.py --row 0 --debug
```

7. Plaintext sanity check

```powershell
python debug_plain_score.py
```

---

## ⚠️ Security & Limitations

- CKKS is approximate: reconstructed logits may have small error; median scaling is a heuristic (acceptable for demo, not production-grade FHE pipeline).
- Homomorphic addition of all contributions server-side is omitted (would require managing rescaling & relinearization for packed vectors). Extension idea: pack all features then use rotations + additions.
- AES key management is simplistic (stored in plain text file). In production use a KMS / HSM.
- Logistic regression weights are sent in plaintext to server implicitly (server code loads them locally) — model confidentiality is not preserved here.
- No differential privacy or protection against membership inference on model.

---

## 🚀 Potential Extensions

- Perform encrypted summation and return single ciphertext (privacy improvement for contribution pattern).
- Add support for polynomial sigmoid approximation homomorphically for fully encrypted probability before client decrypts.
- Integrate AAD in AES stage (schema hash already scaffolded).
- Add integrity verification / signatures for model params.
- Multi-row batch encryption & vector packing to exploit CKKS SIMD.
- Replace median scaling with direct `global_scale` tracking & controlled rescale sequence.

---

## 📁 Directory Tree (Relevant)

```
crypto/
  train_and_export_model.py
  main.py
  app.py
  debug_plain_score.py
  aes_decrypt_all_in_RAM.py
  client_ckks_encrypt_scalars.py
  server_he_infer_scalars.py
  client_decrypt_and_label.py
  models/
    logreg.joblib
    scaler.joblib
    model_params.json
  scripts/
    encodingToAES256.py
    encrypted_dataset.csv
    decrypted_dataset.csv
    nonces_cells.csv
    aes_key.txt
  he_out/
    ctx_pub.bin
    ctx_with_sk.bin
    enc_x_scalars.pkl
    enc_mul_scalars.pkl
```

---

## ✅ Summary

This repository showcases a didactic, modular pipeline where: (1) data can be encrypted at rest with AES‑256‑GCM, (2) a logistic regression model is trained and exported, and (3) a hybrid homomorphic evaluation performs only the feature\*weight multiplications server-side under CKKS, leaving aggregation and sigmoid to the client after decrypting. The README documents each component, data flow, math, and execution steps, and outlines realistic improvements for a production‑ready secure inference system.

Feel free to open issues / extend the pipeline.
