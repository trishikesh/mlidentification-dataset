import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

SEED = 1337
OUT_DIR = "outputs"
CSV_PATH = os.path.join(OUT_DIR, "dataset_features.csv")

def build_preprocessor(X: pd.DataFrame):
    # detect optional categorical columns
    cat_cols = []
    if "content_kind" in X.columns:
        cat_cols.append("content_kind")

    # numeric columns: everything except categorical
    num_cols = [c for c in X.columns if c not in cat_cols]

    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

    return ColumnTransformer(transformers, remainder="drop")

def main():
    df = pd.read_csv(CSV_PATH)

    # CRITICAL FIX: convert all column names to strings to avoid sklearn mixed-type error
    df.columns = df.columns.astype(str)

    if "label" not in df.columns:
        raise ValueError("dataset_features.csv must contain a 'label' column")

    y = df["label"].values
    X = df.drop(columns=["label"]).copy()

    # Some of your older CSVs may not include these; this is fine.
    # If present, keep them as features (they help classification a LOT).
    # (No need to drop size_kb unless you want an ablation later.)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    pre = build_preprocessor(X)

    # Strong baselines
    candidates = {
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=1200, random_state=SEED, n_jobs=-1,
            max_features="sqrt", bootstrap=False
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=1200, random_state=SEED, n_jobs=-1,
            max_features="sqrt"
        ),
        "SVC_RBF": SVC(kernel="rbf", C=30, gamma="scale"),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(512,256), alpha=1e-4,
            max_iter=500, random_state=SEED
        )
    }

    results = []

    # Quick tuning (only for the two most important models)
    tuned_models = {}

    # 1) ExtraTrees quick search
    et_pipe = Pipeline([("pre", pre), ("clf", ExtraTreesClassifier(random_state=SEED, n_jobs=-1))])
    et_space = {
        "clf__n_estimators": [600, 1000, 1400],
        "clf__max_features": ["sqrt", "log2", None],
        "clf__min_samples_split": [2, 4, 8],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__bootstrap": [False, True],
    }
    et_search = RandomizedSearchCV(
        et_pipe, et_space, n_iter=15, scoring="f1_macro",
        cv=4, random_state=SEED, n_jobs=-1, verbose=1
    )
    et_search.fit(X_train, y_train)
    tuned_models["ExtraTrees_Tuned"] = et_search.best_estimator_

    # 2) SVC quick search
    svc_pipe = Pipeline([("pre", pre), ("clf", SVC(kernel="rbf"))])
    svc_space = {
        "clf__C": [3, 10, 30, 100],
        "clf__gamma": ["scale", 0.01, 0.03, 0.1],
    }
    svc_search = RandomizedSearchCV(
        svc_pipe, svc_space, n_iter=8, scoring="f1_macro",
        cv=4, random_state=SEED, n_jobs=-1, verbose=1
    )
    svc_search.fit(X_train, y_train)
    tuned_models["SVC_Tuned"] = svc_search.best_estimator_

    # Evaluate tuned models + baselines
    all_models = {}
    all_models.update(candidates)
    all_models.update(tuned_models)

    for name, model in all_models.items():
        if not isinstance(model, Pipeline):
            model = Pipeline([("pre", pre), ("clf", model)])

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rep = classification_report(y_test, pred, output_dict=True, zero_division=0)

        results.append({
            "model": name,
            "accuracy": rep["accuracy"],
            "precision_macro": rep["macro avg"]["precision"],
            "recall_macro": rep["macro avg"]["recall"],
            "f1_macro": rep["macro avg"]["f1-score"],
        })

        print("\n=== ", name, " ===")
        print(classification_report(y_test, pred, zero_division=0))

    metrics = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "metrics_models.csv")
    metrics.to_csv(out_path, index=False)
    print("\n[+] Saved", out_path)
    print(metrics.head(10))

if __name__ == "__main__":
    main()
