import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

SEED = 1337
OUT_DIR = "outputs"
CSV_PATH = os.path.join(OUT_DIR, "dataset_features.csv")

def make_ohe():
    # sklearn version-safe dense one-hot
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a COPY of df with additional derived features.
    No fake rows, no fabrication. Just legal transformations.
    Also avoids fragmentation by adding new columns via concat once.
    """
    df = df.copy()
    df.columns = df.columns.astype(str)

    # expected: base features at 0..8, hist at 9..264
    base_cols = [str(i) for i in range(9)]
    hist_cols = [str(i) for i in range(9, 265)]

    # coerce numeric safety
    for c in base_cols + hist_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "size_kb" in df.columns:
        df["size_kb"] = pd.to_numeric(df["size_kb"], errors="coerce")

    eps = 1e-12

    # Base columns (as you encoded them)
    ct_len = df["0"]
    enc_ns_med = df["1"]
    enc_ns_per_byte = df["2"]

    pt_len = df["size_kb"] * 1024.0 if "size_kb" in df.columns else np.nan

    # Histogram matrix
    H = df[hist_cols].to_numpy(dtype=np.float64)
    H = np.clip(H, 0.0, 1.0)
    H = H / (H.sum(axis=1, keepdims=True) + eps)

    U = 1.0 / 256.0
    diff = H - U

    # distribution stats
    H_sorted = np.sort(H, axis=1)
    idx = np.arange(256, dtype=np.float64)

    mean = (H * idx).sum(axis=1)
    var = (H * (idx - mean[:, None]) ** 2).sum(axis=1)
    std = np.sqrt(var + eps)
    skew = (H * (idx - mean[:, None]) ** 3).sum(axis=1) / (std ** 3 + eps)
    kurt = (H * (idx - mean[:, None]) ** 4).sum(axis=1) / (std ** 4 + eps)

    # Create derived feature dict (ONE SHOT to avoid fragmentation)
    new = {}

    new["log_ct_len"] = np.log(ct_len + 1.0)
    new["log_enc_ns"] = np.log(enc_ns_med + 1.0)
    new["log_ns_per_byte"] = np.log(enc_ns_per_byte + 1.0)

    if "size_kb" in df.columns:
        pad_overhead = (ct_len - pt_len).fillna(0.0)
        new["pad_overhead"] = pad_overhead
        new["overhead_ratio"] = (pad_overhead / (pt_len + eps)).fillna(0.0)
    else:
        new["pad_overhead"] = 0.0
        new["overhead_ratio"] = 0.0

    new["ct_mod8"] = (ct_len % 8).fillna(0.0)
    new["ct_mod16"] = (ct_len % 16).fillna(0.0)
    new["ct_mod32"] = (ct_len % 32).fillna(0.0)

    new["ns_per_ct_byte"] = (enc_ns_med / (ct_len + eps)).fillna(0.0)

    new["hist_l1_uniform"] = np.sum(np.abs(diff), axis=1)
    new["hist_l2_uniform"] = np.sqrt(np.sum(diff * diff, axis=1))
    new["gini_impurity"] = 1.0 - np.sum(H * H, axis=1)
    new["kl_to_uniform"] = np.sum(H * np.log((H + eps) / U), axis=1)

    new["top1_mass"] = H_sorted[:, -1]
    new["top4_mass"] = np.sum(H_sorted[:, -4:], axis=1)
    new["top16_mass"] = np.sum(H_sorted[:, -16:], axis=1)

    new["byte_mean"] = mean
    new["byte_std"] = std
    new["byte_skew"] = skew
    new["byte_kurt"] = kurt

    new["bins_gt_1pct"] = (H > 0.01).sum(axis=1)
    new["bins_gt_2pct"] = (H > 0.02).sum(axis=1)

    new_df = pd.DataFrame(new)
    out = pd.concat([df, new_df], axis=1)

    # defragment fully
    out = out.copy()
    return out

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = []
    if "content_kind" in X.columns:
        cat_cols.append("content_kind")

    num_cols = [c for c in X.columns if c not in cat_cols]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", make_ohe(), cat_cols),
        ],
        remainder="drop"
    )

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.astype(str)

    if "label" not in df.columns:
        raise ValueError("CSV must contain 'label' column")

    # Create enriched copy
    df_enriched = add_derived_features(df)
    enriched_path = os.path.join(OUT_DIR, "dataset_features_enriched.csv")
    df_enriched.to_csv(enriched_path, index=False)
    print(f"[+] Saved {enriched_path}")

    # Prepare X/y
    df_enriched.columns = df_enriched.columns.astype(str)
    y = df_enriched["label"].values
    X = df_enriched.drop(columns=["label"]).copy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    pre = build_preprocessor(X)

    # Force float after preprocessing (fixes MLP isnan crash)
    cast_float = FunctionTransformer(lambda a: a.astype(np.float64), accept_sparse=False)

    model_spaces = {
        "RandomForest_Tuned": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {
                "clf__n_estimators": [600, 1000, 1400],
                "clf__max_depth": [None, 20, 40, 60],
                "clf__max_features": ["sqrt", "log2", None],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__min_samples_split": [2, 4, 8],
            }
        ),
        "ExtraTrees_Tuned": (
            ExtraTreesClassifier(random_state=SEED, n_jobs=-1),
            {
                "clf__n_estimators": [800, 1200, 1600],
                "clf__max_features": ["sqrt", "log2", None],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__min_samples_split": [2, 4, 8],
                "clf__bootstrap": [False, True],
            }
        ),
        "SVC_Tuned_PCA": (
            SVC(kernel="rbf"),
            {
                "pca__n_components": [30, 50, 80, 120],
                "clf__C": [3, 10, 30, 100],
                "clf__gamma": ["scale", 0.01, 0.03, 0.1],
            }
        ),
        "KNN_Tuned_PCA": (
            KNeighborsClassifier(),
            {
                "pca__n_components": [30, 50, 80, 120],
                "clf__n_neighbors": [3, 5, 7, 11, 15],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            }
        ),
        "DecisionTree_Tuned": (
            DecisionTreeClassifier(random_state=SEED),
            {
                "clf__max_depth": [None, 10, 20, 40],
                "clf__min_samples_split": [2, 4, 8, 16],
                "clf__min_samples_leaf": [1, 2, 4, 8],
                "clf__max_features": ["sqrt", "log2", None],
            }
        ),
        "MLP_Tuned_PCA": (
            MLPClassifier(max_iter=700, random_state=SEED, early_stopping=False),
            {
                "pca__n_components": [50, 80, 120],
                "clf__hidden_layer_sizes": [(256,128), (512,256), (512,256,128)],
                "clf__alpha": [1e-5, 1e-4, 1e-3],
                "clf__learning_rate_init": [1e-3, 5e-4, 2e-4],
                "clf__batch_size": [64, 128, 256],
            }
        ),
    }

    results = []

    for name, (clf, space) in model_spaces.items():
        use_pca = name.endswith("_PCA")

        if use_pca:
            pipe = Pipeline([("pre", pre), ("pca", PCA(random_state=SEED)), ("cast", cast_float), ("clf", clf)])
        else:
            pipe = Pipeline([("pre", pre), ("cast", cast_float), ("clf", clf)])

        search = RandomizedSearchCV(
            pipe,
            space,
            n_iter=16,
            scoring="f1_macro",
            cv=4,
            random_state=SEED,
            n_jobs=-1,
            verbose=1,
            error_score="raise",  # show real errors immediately
        )

        search.fit(X_train, y_train)
        best = search.best_estimator_
        pred = best.predict(X_test)

        rep = classification_report(y_test, pred, output_dict=True, zero_division=0)

        results.append({
            "model": name,
            "accuracy": rep["accuracy"],
            "precision_macro": rep["macro avg"]["precision"],
            "recall_macro": rep["macro avg"]["recall"],
            "f1_macro": rep["macro avg"]["f1-score"],
            "best_params": str(search.best_params_),
        })

        print(f"\n=== {name} (best) ===")
        print(classification_report(y_test, pred, zero_division=0))

    out = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    out_path = os.path.join(OUT_DIR, "metrics_models_enhanced.csv")
    out.to_csv(out_path, index=False)

    print(f"\n[+] Saved {out_path}")
    print(out[["model","accuracy","precision_macro","recall_macro","f1_macro"]].head(20))

if __name__ == "__main__":
    main()
