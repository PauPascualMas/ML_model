#!/usr/bin/env python
# coding: utf-8

"""
Finalize CatBoost model from merged Optuna study
- Train final model
- Evaluate on test set
- Plot TOP-10 PR curves
"""

import os
import argparse
import json
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from catboost import CatBoostClassifier
import optuna
import joblib

# ===============================
# CLI
# ===============================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, required=True)
    parser.add_argument("--storage", type=str, required=True)
    return parser.parse_args()

args = parse_args()
np.random.seed(args.seed)

# ===============================
# Paths
# ===============================
MODEL_DIR = os.path.join(args.output, "model")
PLOT_DIR = os.path.join(args.output, "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ===============================
# Load data
# ===============================
df = pd.read_csv(args.data)
TARGET = args.target

NONTARGETCOLS = [
    "record_id", "episode_id", "fecha_ingreso", "fecha_alta",
    "fecha_mortalidad", "mortalidad", "dias_hemocultivo_mortalidad",
    "microorganismos_dict", "fechas_dict", "especimen_dict",
    "duracion_UCI", "antimicrobiano_list", "cmi_list",
    "interpretacion_list", "antimicrobiano_family_list"
]

df = df.drop(columns=[c for c in NONTARGETCOLS if c in df.columns])

NA_THRESHOLD = 5
na_pct = df.isna().mean() * 100
df = df.drop(columns=na_pct[na_pct > NA_THRESHOLD].index.tolist())

X = df.drop(columns=[TARGET])
y = df[TARGET]

prevalence = y.mean()
imbalance_ratio = (y == 0).sum() / (y == 1).sum()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=args.seed
)

# ===============================
# Load merged study
# ===============================
study = optuna.load_study(
    study_name=args.study_name,
    storage=args.storage
)

# ===============================
# Train final model
# ===============================
best_params = study.best_trial.params

final_params = {
    **best_params,
    "iterations": 2000,
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "class_weights": [1, imbalance_ratio],
    "random_seed": args.seed,
    "verbose": False,
    "allow_writing_files": False
}

model = CatBoostClassifier(**final_params)
model.fit(X_train, y_train)

joblib.dump(model, f"{MODEL_DIR}/catboost_model.pkl")
joblib.dump(X.columns.tolist(), f"{MODEL_DIR}/features.pkl")

# ===============================
# Test performance
# ===============================
y_pred = model.predict_proba(X_test)[:, 1]

roc_test = roc_auc_score(y_test, y_pred)
pr_test = average_precision_score(y_test, y_pred)

# ===============================
# Save metadata
# ===============================
metadata = {
    "date": date.today().isoformat(),
    "n_trials": len(study.trials),
    "best_pr_auc_cv": study.best_trial.user_attrs["pr_auc"],
    "best_roc_auc_cv": study.best_trial.user_attrs["roc_auc"],
    "test_pr_auc": pr_test,
    "test_roc_auc": roc_test,
    "prevalence": prevalence
}

with open(f"{MODEL_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# ===============================
# Plot TOP-10 PR curves
# ===============================
top_trials = sorted(
    study.trials,
    key=lambda t: t.user_attrs.get("pr_auc", 0),
    reverse=True
)[:10]

colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(top_trials)))

plt.figure(figsize=(7, 6))

for i, (trial, color) in enumerate(zip(top_trials[::-1], colors)):
    params = {
        **trial.params,
        "iterations": 2000,
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "class_weights": [1, imbalance_ratio],
        "random_seed": args.seed,
        "verbose": False,
        "allow_writing_files": False
    }

    m = CatBoostClassifier(**params)
    m.fit(X_train, y_train)

    y_pred = m.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred)

    plt.plot(
        recall,
        precision,
        color=color,
        linewidth=1.5,
        label=f"Rank {10-i}"
    )

plt.hlines(
    prevalence, 0, 1,
    linestyles="dashed",
    colors="grey",
    label=f"Baseline (prev={prevalence:.2f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Top-10 CatBoost models (PR curves)")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/top10_pr_curves.png")
plt.close()
