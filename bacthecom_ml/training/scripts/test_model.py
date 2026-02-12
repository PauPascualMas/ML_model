#!/usr/bin/env python
# coding: utf-8

"""
Finalize CatBoost model from best Optuna trial

- Train final model
- Evaluate on test set
- Threshold optimization (F1)
- Save metrics
- Plot ROC, PR, confusion matrix
- Compute and plot SHAP feature importance
"""

# ===============================
# Imports
# ===============================
import os, time, json
from datetime import date
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # HPC-safe
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)

from catboost import CatBoostClassifier, Pool
import optuna
import shap
import joblib

# ===============================
# CLI
# ===============================
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, required=True)
    parser.add_argument("--storage", type=str, required=True)
    return parser.parse_args()

args = parse_args()
np.random.seed(args.seed)

# ===============================
# Helper logging
# ===============================
def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ===============================
# Paths
# ===============================
MODEL_DIR = args.model_dir
RESULTS_DIR = args.results_dir
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ===============================
# Load data
# ===============================
df = pd.read_csv(args.data)
TARGET = args.target

NONTARGETCOLS = [
    "record_id", "episode_id", "fecha_ingreso", "fecha_alta", "mortalidad",
    "microorganismos_dict", "fechas_dict", "especimen_dict",
    "antimicrobiano_list", "cmi_list", "interpretacion_list",
    "antimicrobiano_family_list"
]

df_ = df.drop(columns=NONTARGETCOLS)
X = df_.drop(columns=[TARGET])
y = df_[TARGET]

FEATURES = X.columns.tolist()

imbalance_ratio = (y == 0).sum() / (y == 1).sum()
prevalence = y.mean()

# ===============================
# Train / test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=args.seed
)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# ===============================
# Load Optuna study
# ===============================
log("Loading Optuna study")
study = optuna.load_study(
    study_name=args.study_name,
    storage=args.storage
)

best_trial = study.best_trial
best_params = best_trial.params.copy()

pos_w = best_params.pop("pos_weight", imbalance_ratio)
best_params["class_weights"] = [1.0, pos_w]

best_iteration = best_trial.user_attrs.get("best_iteration", 750)

# ===============================
# Train final model
# ===============================
final_params = {
    **best_params,
    "iterations": best_iteration,
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "random_seed": args.seed,
    "verbose": False,
    "allow_writing_files": False
}

log("Training final CatBoost model")
model = CatBoostClassifier(**final_params)
model.fit(X_train, y_train)

joblib.dump(model, f"{MODEL_DIR}/catboost_model.pkl")
joblib.dump(FEATURES, f"{MODEL_DIR}/features.pkl")

# ===============================
# SHAP Feature Importance (CORRECT)
# ===============================
log("Computing SHAP values")

# Subsample for speed if needed
if len(X_train) > 2000:
    X_shap = X_train.sample(2000, random_state=args.seed)
    y_shap = y_train.loc[X_shap.index]
else:
    X_shap = X_train.copy()
    y_shap = y_train.copy()

shap_pool = Pool(X_shap, y_shap)

shap_values = model.get_feature_importance(
    shap_pool,
    type="ShapValues"
)

# Remove expected value column
shap_values = shap_values[:, :-1]

shap_df = pd.DataFrame(shap_values, columns=FEATURES)

shap_importance = (
    shap_df.abs()
    .mean()
    .sort_values(ascending=False)
)

shap_importance.to_csv(
    f"{RESULTS_DIR}/shap_feature_importance.csv"
)

# SHAP plots
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
plt.savefig(f"{PLOTS_DIR}/shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values, X_shap, plot_type="bar",
    show=False, max_display=20
)
plt.savefig(f"{PLOTS_DIR}/shap_bar.png", dpi=300, bbox_inches="tight")
plt.close()

log("Saved SHAP outputs")

# ===============================
# Test evaluation
# ===============================
y_prob = model.predict_proba(X_test)[:, 1]

pr_test = average_precision_score(y_test, y_prob)
roc_test = roc_auc_score(y_test, y_prob)

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

y_pred = (y_prob >= best_threshold).astype(int)

precision_at_best = precision_score(y_test, y_pred, zero_division=0)
recall_at_best = recall_score(y_test, y_pred, zero_division=0)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (thr={best_threshold:.3f})")
plt.savefig(f"{PLOTS_DIR}/confusion_matrix.png")
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC={roc_test:.3f}")
plt.plot([0,1],[0,1],'--')
plt.legend()
plt.savefig(f"{PLOTS_DIR}/roc_curve.png")
plt.close()

plt.plot(recall, precision, label=f"PR-AUC={pr_test:.3f}")
plt.hlines(prevalence, 0, 1, linestyles="dashed")
plt.legend()
plt.savefig(f"{PLOTS_DIR}/pr_curve.png")
plt.close()

# ===============================
# Save metadata
# ===============================
metadata = {
    "date": date.today().isoformat(),
    "n_trials": len(study.trials),
    "cv_pr_auc": best_trial.user_attrs.get("pr_auc"),
    "cv_roc_auc": best_trial.user_attrs.get("roc_auc"),
    "test_pr_auc": float(pr_test),
    "test_roc_auc": float(roc_test),
    "best_threshold": float(best_threshold),
    "test_precision_at_best_threshold": float(precision_at_best),
    "test_recall_at_best_threshold": float(recall_at_best),
    "prevalence": float(prevalence),
    "n_features": len(FEATURES),
    "confusion_matrix": cm.tolist()
}


with open(f"{MODEL_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

log("DONE — model, metrics, plots, SHAP saved")
