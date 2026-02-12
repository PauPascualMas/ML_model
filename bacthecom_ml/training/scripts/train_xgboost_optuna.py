#!/usr/bin/env python
# coding: utf-8

"""
XGBoost hyperparameter optimization (HPC + Slurm array)

- Optuna ONLY
- Shared RDB storage
- 70/30 train/test split
- 4-fold CV only on training set
- PR-AUC optimized for imbalanced target
"""

# ===============================
# Imports
# ===============================
import os
import json
import argparse
from warnings import filterwarnings
filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ===============================
# CLI arguments
# ===============================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
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
JOBID = os.environ.get("SLURM_JOB_ID", 0)
array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
seed = args.seed + array_id
np.random.seed(seed)

# ===============================
# CPU usage
# ===============================
N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

# ===============================
# Directories
# ===============================
os.makedirs(args.model_dir, exist_ok=True)

RUN_DIR = os.path.join(args.results_dir, f"train_{JOBID}")
os.makedirs(RUN_DIR, exist_ok=True)

PARAMS_PATH = os.path.join(RUN_DIR, "best_params.json")
METRICS_PATH = os.path.join(RUN_DIR, "best_metrics.json")
OPTUNA_CSV_PATH = os.path.join(RUN_DIR, f"optuna_trials_array{array_id}.csv")

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

# ===============================
# Imbalance ratio
# ===============================
imbalance_ratio = (y == 0).sum() / (y == 1).sum()

# ===============================
# Train / test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=seed
)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# ===============================
# CV on training set
# ===============================
cv = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=seed
)

# ===============================
# Optuna objective
# ===============================
best_fold_pr = 0.0

def objective(trial):
    global best_fold_pr

    params = {
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 6),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 10, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", imbalance_ratio * 0.75, imbalance_ratio * 1.25),
        "objective": "binary:logistic",
        "eval_metric": "aucpr",   # PR-AUC
        "use_label_encoder": False,
        "random_state": seed,
        "n_jobs": N_CPUS,
    }

    pr_scores = []
    roc_scores = []

    for fold_idx, (tr_idx, va_idx) in enumerate(
        cv.split(X_train, y_train), start=1
    ):
        X_tr, X_va = X_train.values[tr_idx], X_train.values[va_idx]
        y_tr, y_va = y_train.values[tr_idx], y_train.values[va_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            early_stopping_rounds=100,
            verbose=False
        )

        y_pred = model.predict_proba(X_va)[:, 1]

        pr = average_precision_score(y_va, y_pred)
        roc = roc_auc_score(y_va, y_pred)

        pr_scores.append(pr)
        roc_scores.append(roc)

        if pr > best_fold_pr:
            best_fold_pr = pr
            print(
                f"[Trial {trial.number} | Fold {fold_idx}] "
                f"PR-AUC={pr:.4f}, ROC-AUC={roc:.4f}",
                flush=True
            )

    mean_pr = float(np.mean(pr_scores))
    mean_roc = float(np.mean(roc_scores))

    trial.set_user_attr("pr_auc", mean_pr)
    trial.set_user_attr("roc_auc", mean_roc)

    return mean_pr

# ===============================
# Optuna study
# ===============================
study = optuna.create_study(
    study_name=args.study_name,
    storage=args.storage,
    sampler=optuna.samplers.TPESampler(
        n_startup_trials=10,
        n_ei_candidates=24
    ),
    direction="maximize",
    load_if_exists=True
)

# Run optimization
study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

# ===============================
# Evaluate best trial on test set
# ===============================
best_params = study.best_trial.params.copy()

# scale_pos_weight already applied in params
final_model = xgb.XGBClassifier(
    **best_params,
    n_estimators=best_params.get("n_estimators", 1000),
    objective="binary:logistic",
    eval_metric="aucpr",
    use_label_encoder=False,
    random_state=seed,
    n_jobs=N_CPUS,
)

final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)

y_test_pred = final_model.predict_proba(X_test)[:, 1]
pr_test = average_precision_score(y_test, y_test_pred)
roc_test = roc_auc_score(y_test, y_test_pred)

# ===============================
# Save Optuna trials
# ===============================
study_df = study.trials_dataframe()
study_df.to_csv(OPTUNA_CSV_PATH, index=False)
print(f"Saved Optuna trials to: {OPTUNA_CSV_PATH}")

# ===============================
# Save best parameters
# ===============================
best_params_to_save = study.best_trial.params.copy()
best_params_to_save.update({
    "n_estimators": final_model.n_estimators,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "random_state": seed,
    "n_jobs": N_CPUS,
    "array_task_id": array_id,
    "job_id": JOBID
})

with open(PARAMS_PATH, "w") as f:
    json.dump(best_params_to_save, f, indent=4)
print(f"Saved best hyperparameters to: {PARAMS_PATH}")

# ===============================
# Save best metrics
# ===============================
best_metrics = {
    "study_name": args.study_name,
    "job_id": JOBID,
    "array_task_id": array_id,
    "best_trial_number": study.best_trial.number,
    "cv_pr_auc": study.best_trial.user_attrs.get("pr_auc"),
    "cv_roc_auc": study.best_trial.user_attrs.get("roc_auc"),
    "test_pr_auc": pr_test,
    "test_roc_auc": roc_test,
    "seed": seed,
    "timestamp": pd.Timestamp.now().isoformat()
}

with open(METRICS_PATH, "w") as f:
    json.dump(best_metrics, f, indent=4)
print(f"Saved best metrics to: {METRICS_PATH}")

# ===============================
# Print summary
# ===============================
print("\n" + "=" * 80)
print("BEST TRIAL SUMMARY")
print("=" * 80)

print(f"Date & Time     : {pd.Timestamp.now()}")
print(f"Study name      : {args.study_name}")
print(f"Best trial ID   : {study.best_trial.number}")
print(f"CV PR-AUC       : {study.best_trial.user_attrs.get('pr_auc'):.4f}")
print(f"CV ROC-AUC      : {study.best_trial.user_attrs.get('roc_auc'):.4f}")
print(f"Test PR-AUC     : {pr_test:.4f}")
print(f"Test ROC-AUC    : {roc_test:.4f}")
print(f"Seed            : {seed}")
print(f"Array task ID   : {array_id}")

print("-" * 80)
print("Best hyperparameters:")
for k, v in study.best_trial.params.items():
    print(f"  {k}: {v}")

print("=" * 80 + "\n")
