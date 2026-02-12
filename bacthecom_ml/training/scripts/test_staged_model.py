#!/usr/bin/env python
# coding: utf-8

"""
Evaluate STAGED CatBoost models (NO TRAINING)

Stage 1: Any death (all patients)
Stage 2: 30-day death (ONLY among patients who died)

- Correct conditional evaluation
- Correct PR baselines
- ROC, PR, confusion matrix
- SHAP for Stage 2
"""

# ===============================
# Imports
# ===============================
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)

from catboost import Pool
import shap
import joblib


# ===============================
# Helpers
# ===============================
def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ===============================
# Paths
# ===============================
MODEL_DIR = "../model"
RESULTS_DIR = "../results/testing/"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ===============================
# Load models & features
# ===============================
log("Loading trained models")

model_stage1 = joblib.load(f"{MODEL_DIR}/stage1_any_death_model.pkl")
model_stage2 = joblib.load(f"{MODEL_DIR}/stage2_30day_model.pkl")

features_stage2 = joblib.load(f"{MODEL_DIR}/stage2_features.pkl")

# ===============================
# Load data
# ===============================
df = pd.read_csv("../data/preprocessed_db.csv")

DROP_COLS = [
    "record_id", "episode_id", "fecha_ingreso", "fecha_alta",
    "microorganismos_dict", "fechas_cultivo_dict", "especimen_dict",
    "antimicrobiano_list", "cmi_list", "interpretacion_list",
    "antimicrobiano_family_list", 
]

df = df.drop(columns=DROP_COLS)

y1 = df["mortalidad"]          # any death
y2 = df["mortalidad_30_dias"]          # 30-day death
X  = df.drop(columns=["mortalidad", "mortalidad_30_dias"])

# ===============================
# Train / test split (MATCH TRAINING)
# ===============================
X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    X, y1, y2,
    test_size=0.35,
    stratify=y1,                 # MUST match training
    random_state=42
)

X_test = X_test.astype(np.float32)

# ===============================
# Stage 1 inference (ANY death)
# ===============================
log("Running Stage 1 inference")

p1_test = model_stage1.predict_proba(X_test)[:, 1]

pr1 = average_precision_score(y1_test, p1_test)
roc1 = roc_auc_score(y1_test, p1_test)

# ===============================
# Stage 2 population (ONLY deaths)
# ===============================
mask_stage2 = y1_test == 1

X_test_s2 = X_test.loc[mask_stage2].copy()
y2_test_s2 = y2_test.loc[mask_stage2].copy()

# Prevalence among deaths
prevalence_30d_conditional = y2_test_s2.mean()

# Add Stage 1 prediction as feature
X_test_s2["pred_stage1"] = p1_test[mask_stage2]

# Align feature order
X_test_s2 = X_test_s2[features_stage2]

# ===============================
# Stage 2 inference (30-day death)
# ===============================
log("Running Stage 2 inference")

p2_test = model_stage2.predict_proba(X_test_s2)[:, 1]

pr2 = average_precision_score(y2_test_s2, p2_test)
roc2 = roc_auc_score(y2_test_s2, p2_test)

# ===============================
# Threshold optimization (Stage 2)
# ===============================
precision, recall, thresholds = precision_recall_curve(y2_test_s2, p2_test)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

best_idx = np.argmax(f1)
best_thr = thresholds[best_idx]

y_pred = (p2_test >= best_thr).astype(int)

precision_best = precision_score(y2_test_s2, y_pred, zero_division=0)
recall_best = recall_score(y2_test_s2, y_pred, zero_division=0)

cm = confusion_matrix(y2_test_s2, y_pred)

# ===============================
# Plots
# ===============================
log("Generating plots")

# Confusion matrix
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Stage 2 Confusion Matrix (thr={best_thr:.3f})")
plt.savefig(f"{PLOTS_DIR}/confusion_matrix_stage2.png", dpi=300)
plt.close()

# ROC
fpr, tpr, _ = roc_curve(y2_test_s2, p2_test)
plt.plot(fpr, tpr, label=f"AUC={roc2:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("Stage 2 ROC (conditional on death)")
plt.savefig(f"{PLOTS_DIR}/roc_stage2.png", dpi=300)
plt.close()

# PR
plt.plot(recall, precision, label=f"PR-AUC={pr2:.3f}")
plt.hlines(
    prevalence_30d_conditional, 0, 1,
    linestyles="dashed",
    label=f"Baseline={prevalence_30d_conditional:.3f}"
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.title("Stage 2 PR Curve (conditional on death)")
plt.savefig(f"{PLOTS_DIR}/pr_stage2.png", dpi=300)
plt.close()

# ===============================
# SHAP (Stage 2 only)
# ===============================
log("Computing SHAP values (Stage 2)")

shap_pool = Pool(X_test_s2, y2_test_s2)

shap_values = model_stage2.get_feature_importance(
    shap_pool,
    type="ShapValues"
)[:, :-1]

shap.summary_plot(
    shap_values,
    X_test_s2,
    show=False,
    max_display=20
)

plt.savefig(
    f"{PLOTS_DIR}/shap_stage2_summary.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ===============================
# Save metrics
# ===============================
metrics = {
    "stage1": {
        "test_pr_auc": float(pr1),
        "test_roc_auc": float(roc1),
        "prevalence": float(y1_test.mean())
    },
    "stage2": {
        "test_pr_auc": float(pr2),
        "test_roc_auc": float(roc2),
        "conditional_prevalence_30d": float(prevalence_30d_conditional),
        "best_threshold": float(best_thr),
        "precision_at_best": float(precision_best),
        "recall_at_best": float(recall_best),
        "confusion_matrix": cm.tolist(),
        "n_test_samples": int(len(y2_test_s2))
    }
}

with open(f"{RESULTS_DIR}/metrics_stage2.json", "w") as f:
    json.dump(metrics, f, indent=2)

log("DONE — evaluation, plots, SHAP, metrics saved")
