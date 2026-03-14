# debug_plain_score.py
import json, numpy as np, pandas as pd
from pathlib import Path

CSV = r".\scripts\decrypted_dataset.csv"
PARAMS = r".\models\model_params.json"
ROW = 0  # same row you used

mp = json.load(open(PARAMS, "r"))
feat = mp["feature_names"]
mu   = np.array(mp["scaler_mean"], dtype=float)
sig  = np.array(mp["scaler_scale"], dtype=float)
w    = np.array(mp["weights"], dtype=float)
b    = float(mp["bias"])

df = pd.read_csv(CSV)
for c in ["id","Unnamed: 32","diagnosis"]:
    if c in df.columns: df = df.drop(columns=[c])

# enforce feature order
x = df.loc[ROW, feat].astype(float).values

# guard against zeros in sigma
eps = 1e-12
sig_safe = np.where(sig == 0, eps, sig)

x_std = (x - mu) / sig_safe
score_plain = float(np.dot(x_std, w) + b)

print(f"Plain score (logit) for row {ROW}: {score_plain:.6f}")
