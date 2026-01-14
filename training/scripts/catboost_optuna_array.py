#!/usr/bin/env python
# coding: utf-8

"""
CatBoost hyperparameter optimization (HPC + Slurm array)

- Optuna ONLY
- Shared RDB storage
- No final model, no plots
"""

# ===============================
# Imports
# ===============================
import os
import argparse
from warnings import filterwarnings
filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from catboost import CatBoostClassifier
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ===============================
# CLI arguments
# ===============================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=100,
                        help="Trials PER array job")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, required=True)
    parser.add_argument("--storage", type=str, required=True)
    return parser.parse_args()

args = parse_args()

# ===============================
# Reproducibility (array-aware)
# ===============================
array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
seed = args.seed + array_id
np.random.seed(seed)

# ===============================
# CPU usage
# ===============================
N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

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

# ===============================
# Drop high-missingness variables
# ===============================
NA_THRESHOLD = 5  # %
na_pct = df.isna().mean() * 100
df = df.drop(columns=na_pct[na_pct > NA_THRESHOLD].index.tolist())

X = df.drop(columns=[TARGET])
y = df[TARGET]

imbalance_ratio = (y == 0).sum() / (y == 1).sum()

# ===============================
# CV
# ===============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# ===============================
# Optuna objective
# ===============================
def objective(trial):

    params = {
        "iterations": 750,
        "early_stopping_rounds": 150,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 6),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 5.0, 80.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 300),
        "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.3, 2.0),
        "bootstrap_type": "Bayesian",
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "class_weights": [
            1.0,
            trial.suggest_float(
                "pos_weight",
                imbalance_ratio * 0.5,
                imbalance_ratio * 1.5
            )
        ],
        "random_seed": seed,
        "thread_count": N_CPUS,
        "verbose": False,
        "allow_writing_files": False,
    }

    pr_scores = []
    roc_scores = []

    for tr_idx, va_idx in cv.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=100,
            verbose=False
        )

        y_pred = model.predict_proba(X_va)[:, 1]
        pr_scores.append(average_precision_score(y_va, y_pred))
        roc_scores.append(roc_auc_score(y_va, y_pred))

    trial.set_user_attr("pr_auc", np.mean(pr_scores))
    trial.set_user_attr("roc_auc", np.mean(roc_scores))

    return np.mean(pr_scores)

# ===============================
# Optuna study (shared)
# ===============================
study = optuna.create_study(
    study_name=args.study_name,
    storage=args.storage,
    direction="maximize",
    load_if_exists=True
)

study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

print(f"[Array task {array_id}] Finished {args.n_trials} trials.")
