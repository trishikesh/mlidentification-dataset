# ML-based Lightweight Cipher Identification

This repository contains the data-processing and model-training workflow for the paper:

**ML-based Lightweight Cipher Identification**

The project builds a machine learning pipeline that identifies which lightweight or conventional cipher produced a ciphertext sample by learning from ciphertext-derived statistical features.

## Project Goal

The core idea is to treat ciphertext as a measurable signal. Instead of relying on cryptanalytic key recovery, this work extracts statistical and timing-oriented features from encrypted outputs and trains supervised classifiers to distinguish between cipher families.

In the current generated dataset, the classification task is a **6-class balanced problem** with the following labels:

- `AES128_CBC`
- `LEA128`
- `MSEA256`
- `RECTANGLE128`
- `SIMON128_128`
- `XTEA`

Each class has `480` samples, for a total of `2880` rows.

## Repository Structure

```text
iotidentification/
├── app.py
├── main.py
├── README.md
└── outputs/
    ├── dataset_features.csv
    ├── dataset_features_enriched.csv
    └── metrics_models.csv
```

## What The Pipeline Does

The repository currently exposes two model workflows:

### 1. Baseline training pipeline (`app.py`)

`app.py` loads the generated feature dataset, prepares numeric and optional categorical inputs, trains several baseline classifiers, performs quick tuning for selected models, and writes a ranked metrics file.

Models evaluated in `app.py`:

- `ExtraTrees`
- `RandomForest`
- `SVC_RBF`
- `MLP`
- `ExtraTrees_Tuned`
- `SVC_Tuned`

The script saves the evaluation summary to:

- `outputs/metrics_models.csv`

### 2. Enriched feature pipeline (`main.py`)

`main.py` extends the baseline workflow by generating additional derived features from the existing dataset before training a broader set of tuned models.

This script:

- Loads `outputs/dataset_features.csv`
- Creates an enriched copy with engineered features
- Saves the enriched dataset to `outputs/dataset_features_enriched.csv`
- Trains tuned models, including PCA-based variants
- Saves the final comparison table to `outputs/metrics_models_enhanced.csv`

Models evaluated in `main.py`:

- `RandomForest_Tuned`
- `ExtraTrees_Tuned`
- `SVC_Tuned_PCA`
- `KNN_Tuned_PCA`
- `DecisionTree_Tuned`
- `MLP_Tuned_PCA`

## Dataset and Feature Layout

The base dataset file is:

- `outputs/dataset_features.csv`

Current dataset summary:

- Total rows: `2880`
- Total columns: `268`
- Feature columns excluding label: `267`
- Number of cipher classes: `6`
- `content_kind` values present: `ascii`, `pattern`, `random`, `zeros`
- `size_kb` values present: `16`, `64`, `256`, `512`, `1024`, `2048`

### Base feature structure

From the code in `main.py`, the dataset is expected to contain:

- Columns `0` to `8`: base extracted features
- Columns `9` to `264`: normalized histogram-style features over 256 bins
- `label`: target cipher class
- `size_kb`: plaintext size category in kilobytes
- `content_kind`: input content pattern category

`main.py` explicitly uses the following base fields while creating derived features:

- `0`: ciphertext length
- `1`: median encryption time in nanoseconds
- `2`: encryption time per byte

### Derived features created in `main.py`

The enriched pipeline adds higher-level descriptors intended to capture ciphertext structure, padding behavior, runtime scaling, and byte-distribution shape.

Derived feature groups include:

- Log-scaled runtime and length features
- Padding overhead and overhead ratio
- Ciphertext length modulo features (`mod 8`, `mod 16`, `mod 32`)
- Runtime per ciphertext byte
- Distance from uniform byte distribution
- Gini impurity and KL divergence to uniform distribution
- Top-bin mass summaries (`top1`, `top4`, `top16`)
- Byte histogram moments: mean, standard deviation, skewness, kurtosis
- Active-bin counts above fixed thresholds

This enriched dataset is stored in:

- `outputs/dataset_features_enriched.csv`

## Preprocessing Strategy

Both scripts use a structured preprocessing pipeline built with scikit-learn.

Common preprocessing steps:

- Numeric features are standardized with `StandardScaler`
- Optional categorical columns such as `content_kind` are encoded with `OneHotEncoder`
- Data is split with `train_test_split(..., test_size=0.2, stratify=y, random_state=1337)`

Additional steps in `main.py`:

- Uses `FunctionTransformer` to cast post-processed arrays to `float64`
- Uses `PCA` for the PCA-based SVC, KNN, and MLP variants
- Uses `RandomizedSearchCV` with macro F1 scoring for tuned model selection

## Recorded Baseline Results

The checked-in evaluation file `outputs/metrics_models.csv` currently reports the following scores:

| Model            | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| ---------------- | -------: | --------------: | -----------: | -------: |
| ExtraTrees_Tuned |   0.9844 |          0.9846 |       0.9844 |   0.9844 |
| RandomForest     |   0.9670 |          0.9676 |       0.9670 |   0.9670 |
| ExtraTrees       |   0.9601 |          0.9611 |       0.9601 |   0.9600 |
| SVC_RBF          |   0.8194 |          0.8303 |       0.8194 |   0.8157 |
| SVC_Tuned        |   0.7760 |          0.7835 |       0.7760 |   0.7696 |
| MLP              |   0.7344 |          0.7335 |       0.7344 |   0.7332 |

### Current best recorded baseline

The best checked-in result is:

- Model: `ExtraTrees_Tuned`
- Accuracy: `98.4375%`
- Macro F1: `98.4418%`

This suggests that tree-based ensemble methods are currently the strongest baseline for this cipher-identification dataset.

## How To Run

Install the required Python packages first:

```bash
pip install numpy pandas scikit-learn
```

Run the baseline workflow:

```bash
python app.py
```

Run the enriched-feature workflow:

```bash
python main.py
```

## Output Files

### Currently present in `outputs/`

- `dataset_features.csv`: base extracted feature dataset
- `dataset_features_enriched.csv`: feature-engineered dataset created by the enriched pipeline
- `metrics_models.csv`: baseline and quick-tuned model comparison from `app.py`

### Generated when `main.py` is executed

- `metrics_models_enhanced.csv`: tuned-model comparison for the enriched pipeline

## Why This Repository Matters For The Paper

This repository operationalizes the paper idea as a reproducible ML workflow:

- It formalizes ciphertext identification as a supervised classification problem
- It converts encrypted outputs into measurable statistical features
- It compares multiple model families under the same split strategy
- It demonstrates that feature engineering materially strengthens the identification pipeline
- It provides a direct bridge between cryptography experimentation and applied machine learning evaluation

## Summary

In short, this project is the implementation layer behind **ML-based Lightweight Cipher Identification**. It contains:

- A generated cipher-feature dataset
- A baseline model benchmark pipeline
- An enriched feature-engineering pipeline
- Stored evaluation results showing strong performance from ensemble tree models

For a paper submission or project report, this repository can be presented as the experimental training and evaluation backend for cipher-family identification from ciphertext statistics.
