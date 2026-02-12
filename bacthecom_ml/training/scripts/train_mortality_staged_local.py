#!/usr/bin/env python
# coding: utf-8

"""
Two-stage CatBoost hyperparameter optimization (HPC + Slurm array)

Stage 1: Predict any death
Stage 2: Predict 30-day death (only for patients predicted as death in stage 1)

- Optuna ONLY
- Shared RDB storage
- 70/30 train/test split
- 4-fold CV only on training set
"""

# ===============================
# Imports
# ===============================
import os
import time
import json
import argparse
from warnings import filterwarnings
filterwarnings("ignore")
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from catboost import CatBoostClassifier
import optuna
import joblib

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ===============================
# Reproducibility (array-aware)
# ===============================
JOBID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

seed = 42
np.random.seed(seed)

# ===============================
# Directories
# ===============================
MODEL_DIR="../model"
os.makedirs(MODEL_DIR, exist_ok=True)

RUN_DIR = os.path.join("../results/training/", f"train_{JOBID}")
os.makedirs(RUN_DIR, exist_ok=True)


PARAMS_PATH = os.path.join(RUN_DIR, f"best_params_array.json")
METRICS_PATH = os.path.join(RUN_DIR, f"best_metrics_array.json")
OPTUNA_CSV_PATH = os.path.join(RUN_DIR, f"optuna_trials_array.csv")


# ===============================
# Load data
# ===============================
df = pd.read_csv("../data/preprocessed_db.csv")

NONTARGETCOLS = [
    "record_id", "episode_id", "fecha_ingreso", "fecha_alta",
    "microorganismos_dict", "fechas_cultivo_dict", "especimen_dict",
    "antimicrobiano_list", "cmi_list", "interpretacion_list",
    "antimicrobiano_family_list", "duracion_UCI"
]

df_ = df.drop(columns=NONTARGETCOLS)

# Stage targets
TARGET1 = "mortalidad"
TARGET2 = "mortalidad_30_dias"

X = df_.drop(columns=[TARGET1, TARGET2])
y1 = df_[TARGET1]  # any death
y2 = df_[TARGET2]  # 30-day death

# ===============================
# Imbalance ratios
# ===============================
imbalance_ratio1 = (y1==0).sum() / (y1==1).sum()
imbalance_ratio2 = (y2==0).sum() / (y2==1).sum()

# ===============================
# Train / test split
# ===============================
X_train, X_test, y_train1, y_test1 = train_test_split(
    X, y1, test_size=0.35, stratify=y1, random_state=seed
)
y_train2 = y2.loc[y_train1.index]
y_test2 = y2.loc[y_test1.index]

X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)

# ===============================
# CV on training set
# ===============================
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)

# ===============================
# Utility: Optuna objective
# ===============================
def make_objective(X_train, y_train, imbalance_ratio):
    best_fold_pr = 0.0
    def objective(trial):
        nonlocal best_fold_pr
        pos_w = trial.suggest_float(
            "pos_weight",                 # <--- parameter name
            imbalance_ratio * 0.75,       # low
            imbalance_ratio * 1.25        # high
        )
        params = {
            "iterations": 750,
            "early_stopping_rounds": 150,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 10, 120, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 30, 150),
            "random_strength": trial.suggest_float("random_strength", 0, 20),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.5, 1.5),
            "bootstrap_type": "Bayesian",
            "loss_function": "Logloss",
            "eval_metric": "PRAUC",
            "class_weights": [1.0, pos_w],
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
        }
        pr_scores, roc_scores = [], []
        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train), start=1):
            X_tr, X_va = X_train.values[tr_idx], X_train.values[va_idx]
            y_tr, y_va = y_train.values[tr_idx], y_train.values[va_idx]

            model = CatBoostClassifier(**params)

            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)

            y_pred = model.predict_proba(X_va)[:,1]

            pr_scores.append(average_precision_score(y_va, y_pred))
            roc_scores.append(roc_auc_score(y_va, y_pred))
            if pr_scores[-1] > best_fold_pr:
                best_fold_pr = pr_scores[-1]
                print(f"[Trial {trial.number} | Fold {fold_idx}] PR={pr_scores[-1]:.4f} ROC={roc_scores[-1]:.4f}", flush=True)
        mean_pr, mean_roc = float(np.mean(pr_scores)), float(np.mean(roc_scores))

        trial.set_user_attr("pr_auc", mean_pr)
        trial.set_user_attr("roc_auc", mean_roc)

        return mean_pr
    return objective

# ===============================
# Stage 1: Any Death
# ===============================
study_stage1 = optuna.create_study(
    study_name=f"bacthecom_anydeath",
    sampler=optuna.samplers.TPESampler(n_startup_trials=100, n_ei_candidates=24),
    direction="maximize",
    load_if_exists=True
)

print("Starting Stage 1 (Any Death) optimization...")
study_stage1.optimize(make_objective(X_train, y_train1, imbalance_ratio1), n_trials=300, n_jobs=1, gc_after_trial=True)

# Best hypeparams
best_params1 = study_stage1.best_trial.params.copy()
pos_w = best_params1.pop("pos_weight", imbalance_ratio1)
best_params1["class_weights"] = [1.0, pos_w]

model_stage1 = CatBoostClassifier(**best_params1, iterations=750,
                                  loss_function="Logloss", eval_metric="PRAUC",
                                  random_seed=seed,
                                  verbose=False, allow_writing_files=False)
model_stage1.fit(X_train, y_train1)

y_test_pred1 = model_stage1.predict_proba(X_test)[:,1]

pr_test1 = average_precision_score(y_test1, y_test_pred1)
roc_test1 = roc_auc_score(y_test1, y_test_pred1)

# ===============================
# Stage 2: 30-Day Death (mask Stage 1 deaths)
# ===============================

# select episodes with deaths, mortality==1
mask_train2 = y_train1 == 1
mask_test2  = y_test1  == 1

X_train2 = X_train[mask_train2].copy()
y_train2_stage = y_train2[mask_train2].copy()
X_test2  = X_test[mask_test2].copy()
y_test2_stage  = y_test2[mask_test2].copy()

# Add Stage 1 prediction as feature
X_train2["pred_stage1"] = model_stage1.predict_proba(X_train[mask_train2])[:,1]
X_test2["pred_stage1"]  = model_stage1.predict_proba(X_test[mask_test2])[:,1]

study_stage2 = optuna.create_study(
    study_name=f"bacthecom_30day",
    sampler=optuna.samplers.TPESampler(n_startup_trials=10, n_ei_candidates=24),
    direction="maximize",
    load_if_exists=True
)

print("Starting Stage 2 (30-Day Death) optimization...")
study_stage2.optimize(make_objective(X_train2, y_train2_stage, imbalance_ratio2), n_trials=300, n_jobs=1, gc_after_trial=True)

best_params2 = study_stage2.best_trial.params.copy()
pos_w2 = best_params2.pop("pos_weight", imbalance_ratio2)
best_params2["class_weights"] = [1.0, pos_w2]

model_stage2 = CatBoostClassifier(**best_params2, iterations=750,
                                  loss_function="Logloss", eval_metric="AUC",
                                  random_seed=seed,
                                  verbose=False, allow_writing_files=False)
model_stage2.fit(X_train2, y_train2_stage)

y_test_pred2 = model_stage2.predict_proba(X_test2)[:,1]

pr_test2 = average_precision_score(y_test2_stage, y_test_pred2)
roc_test2 = roc_auc_score(y_test2_stage, y_test_pred2)

# ===============================
# Save Optuna trials & parameters
# ===============================
study_stage1.trials_dataframe().to_csv(os.path.join(RUN_DIR, f"optuna_stage1_array.csv"), index=False)
study_stage2.trials_dataframe().to_csv(os.path.join(RUN_DIR, f"optuna_stage2_array.csv"), index=False)

best_params_to_save = {
    "stage1": best_params1,
    "stage2": best_params2,
    "job_id": JOBID,
    "seed": seed
}
with open(PARAMS_PATH, "w") as f:
    json.dump(best_params_to_save, f, indent=4)

best_metrics = {
    "pr_test_stage1": pr_test1,
    "roc_test_stage1": roc_test1,
    "pr_test_stage2": pr_test2,
    "roc_test_stage2": roc_test2,
    "job_id": JOBID,
    "seed": seed
}
with open(METRICS_PATH, "w") as f:
    json.dump(best_metrics, f, indent=4)

# ===============================
# Summary
# ===============================
print("\n" + "="*80)
print("BEST TRIALS SUMMARY")
print("="*80)
print(f"Stage 1 (Any Death) PR-AUC: {pr_test1:.4f}, ROC-AUC: {roc_test1:.4f}")
print(f"Stage 2 (30-Day Death) PR-AUC: {pr_test2:.4f}, ROC-AUC: {roc_test2:.4f}")
print("="*80 + "\n")

# ===============================
# SAVE MODELS
# ===============================
# model
joblib.dump(model_stage1, f"{MODEL_DIR}/stage1_any_death_model.pkl")
joblib.dump(model_stage2, f"{MODEL_DIR}/stage2_30day_model.pkl")
# features
joblib.dump(X_train.columns.tolist(), f"{MODEL_DIR}/stage1_features.pkl")
joblib.dump(X_train2.columns.tolist(), f"{MODEL_DIR}/stage2_features.pkl")
# testing
joblib.dump(X_test.index.tolist(), f"{MODEL_DIR}/test_index.pkl")
joblib.dump(y_test_pred1,f"{MODEL_DIR}/stage1_test_pred_proba.pkl")
joblib.dump(X_test2.index.tolist(),f"{MODEL_DIR}/stage2_test_index.pkl")


