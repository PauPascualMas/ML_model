#!/usr/bin/env python
# coding: utf-8

"""
Modelling - BactHeCom
This script contains code for computing and evaluating different model algorithms
to predict 30-day mortality.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

# ------------------------
# Directories
# ------------------------
DATA_DIR = "../data"
RESULTS_DIR = "../results"
os.makedirs(f"{RESULTS_DIR}/model/selection_model", exist_ok=True)

# ------------------------
# Load data
# ------------------------
df = pd.read_csv(f"{DATA_DIR}/preprocessed_df.csv")
print("Dataframe loaded:", df.shape)

# ------------------------
# Target prevalence
# ------------------------
target_col = "mortalidad_30_dias"
mortality_pct = df[target_col].mean() * 100
print(f"Mortality at 30 days: {mortality_pct:.2f}% of total records")

# ------------------------
# Drop non-target columns
# ------------------------
NONTARGETCOLS = [
    "record_id", "episode_id", "fecha_ingreso", "fecha_alta", "fecha_mortalidad", "mortalidad",
    "dias_hemocultivo_mortalidad", "microorganismos_dict", "fechas_dict", "especimen_dict",
    "duracion_UCI", "uci_por_el_episodio", "antimicrobiano_list", "cmi_list",
    "interpretacion_list", "antimicrobiano_family_list"
]

df_ = df.drop(columns=NONTARGETCOLS)

# ------------------------
# Handle missing values
# ------------------------
NA_THRESHOLD = 5
na_tbl = df_.isna().sum().sort_values(ascending=False)
missing_vars = []

for col, n_missing in na_tbl.items():
    pct_missing = n_missing / len(df_) * 100
    if pct_missing > NA_THRESHOLD:
        missing_vars.append(col)
        print(f"{col}: {n_missing} missing values --> {pct_missing:.2f}% NaN")

df_ = df_.drop(columns=missing_vars)
print(f"Dropped {len(missing_vars)} variables with more than {NA_THRESHOLD}% missing values")
print("Dataframe shape after dropping columns:", df_.shape)

# ------------------------
# Prepare features and target
# ------------------------
X = df_.drop(columns=[target_col])
y = df_[target_col]

# ------------------------
# Split train/test
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=99, stratify=y
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ------------------------
# Define models
# ------------------------
models = {
    "Logistic_Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced", random_state=42, max_iter=1000
        ))
    ]),
    "Random_Forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced", random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
        random_state=42
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        class_weight="balanced", random_state=42, verbose=-1
    ),
    "CatBoost": CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        auto_class_weights="Balanced", verbose=0,
        random_state=42, allow_writing_files=False
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=300, learning_rate=0.05, random_state=42
    ),
    "SVC": Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=42))
    ])
}

# ------------------------
# Train models and evaluate
# ------------------------
results = {}
y_pred_probs = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    results[name] = {"ROC-AUC": roc_auc, "PR-AUC": pr_auc}
    y_pred_probs[name] = y_prob

results_df = pd.DataFrame(results)
results_df.to_csv(f"{RESULTS_DIR}/model/selection_model/model_stats.csv")
print("Results:\n", results_df)

# ------------------------
# Plot PR and ROC curves
# ------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Precision-Recall curves
for name, y_prob in y_pred_probs.items():
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ax[0].plot(recall, precision, label=f"{name} (PR-AUC={results[name]['PR-AUC']:.3f})")

baseline = y_test.mean()
ax[0].hlines(baseline, 0, 1, linestyle="dashed", color="grey", label=f"Baseline (prev={baseline:.2f})")
ax[0].set_xlabel("Recall")
ax[0].set_ylabel("Precision")
ax[0].set_title("Precision–Recall Curves")
ax[0].legend()

# ROC curves
for name, y_prob in y_pred_probs.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax[1].plot(fpr, tpr, label=f"{name} (ROC-AUC={results[name]['ROC-AUC']:.3f})")

ax[1].plot([0, 1], [0, 1], '--', color="grey")
ax[1].set_xlabel("False Positive Rate")
ax[1].set_ylabel("True Positive Rate")
ax[1].set_title("ROC Curves")
ax[1].legend()

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/model/selection_model/pr_roc_models_curves.png")
plt.show()