#!/usr/bin/env python
# coding: utf-8

"""
Model Hyperparameter Optimization - BactHeCom
Using Optuna and RandomSearch to fine-tune Random Forest, LightGBM, XGBoost, and CatBoost
to improve 30-day mortality prediction performance.
"""

import os
import json
from datetime import date

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import joblib

from warnings import filterwarnings
filterwarnings("ignore")

# ------------------------
# Directories
# ------------------------
DATA_DIR = "../data"
RESULTS_DIR = "../results"
os.makedirs(f"{RESULTS_DIR}/model/optimization", exist_ok=True)

# ------------------------
# Load data
# ------------------------
df = pd.read_csv(f"{DATA_DIR}/preprocessed_df.csv")
TARGET = "mortalidad_30_dias"
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
missing_vars = [col for col in df_.columns if df_[col].isna().mean()*100 > NA_THRESHOLD]
df_ = df_.drop(columns=missing_vars)
print(f"Dropped {len(missing_vars)} columns with >{NA_THRESHOLD}% missing values")
print("Data shape after cleaning:", df_.shape)

# ------------------------
# Features and target
# ------------------------
X = df_.drop(columns=[TARGET])
y = df_[TARGET]

prevalence = y.mean()
imbalance_ratio = (y==0).sum() / (y==1).sum()
print(f"Mortality prevalence: {prevalence:.2%}, Imbalance ratio: {imbalance_ratio:.2f}")

# ------------------------
# Train/test split
# ------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ------------------------
# CatBoost tuning via Optuna
# ------------------------
best_pr = 0
best_roc = 0

def ctb_objective(trial):
    global best_pr, best_roc
    params = {
        "iterations": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 6),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 5.0, 80.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 300),
        "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.3, 2.0),
        "bootstrap_type": "Bayesian",
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "class_weights": [1, imbalance_ratio],
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val),
              early_stopping_rounds=100, use_best_model=True, verbose=False)
    y_pred = model.predict_proba(X_val)[:,1]
    pr = average_precision_score(y_val, y_pred)
    roc = roc_auc_score(y_val, y_pred)
    trial.set_user_attr("roc_auc", roc)
    trial.set_user_attr("pr_auc", pr)
    if pr > best_pr:
        best_pr = pr
        best_roc = roc
        print(f"New best trial {trial.number}: ROC={roc:.4f}, PR={pr:.4f}")
    return pr

study_ctb = optuna.create_study(direction="maximize")
study_ctb.optimize(ctb_objective, n_trials=100)

# Fit final CatBoost
best_params_ctb = study_ctb.best_trial.params
catboost_params = {
    "iterations": 1000,
    "learning_rate": best_params_ctb["learning_rate"],
    "depth": best_params_ctb["depth"],
    "l2_leaf_reg": best_params_ctb["l2_leaf_reg"],
    "bagging_temperature": best_params_ctb["bagging_temperature"],
    "random_strength": best_params_ctb["random_strength"],
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "class_weights": [1, imbalance_ratio],
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False
}
final_ctb = CatBoostClassifier(**catboost_params)
final_ctb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

joblib.dump(final_ctb, "../../model/catboost_model.pkl")
joblib.dump(X.columns.values, "../../model/features.pkl")
metadata = {
    "model": "Mortality predictor",
    "version": "1.0",
    "date": date.today().strftime("%Y-%m-%d"),
    "auc": best_roc,
    "prevalence": prevalence
}
with open("../../model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

y_pred_ctb = final_ctb.predict_proba(X_test)[:,1]
roc_ctb = roc_auc_score(y_test, y_pred_ctb)
pr_ctb = average_precision_score(y_test, y_pred_ctb)
print(f"CatBoost test ROC-AUC: {roc_ctb:.3f}, PR-AUC: {pr_ctb:.3f}")

print("Hyperparameter optimization completed.")


# ==========================================================

# X_train, X_val, y_train, y_val = train_test_split(
#     X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
# )

# param_dist_rf = {
#     "n_estimators": [500, 800, 1200],
#     "max_depth": [None, 4, 6, 8],
#     "min_samples_leaf": [5, 10, 20, 40],
#     "min_samples_split": [10, 20, 50],
#     "max_features": ["sqrt", 0.3, 0.5],
# }

# rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
# search_rf = RandomizedSearchCV(
#     rf, param_distributions=param_dist_rf, n_iter=50, scoring="average_precision",
#     cv=5, n_jobs=-1, random_state=42, verbose=1
# )
# search_rf.fit(X_train, y_train)

# best_rf = search_rf.best_estimator_
# best_rf.fit(X_train, y_train)
# calibrated_rf = CalibratedClassifierCV(best_rf, method="isotonic", cv=5)
# calibrated_rf.fit(X_train, y_train)

# y_prob_rf = calibrated_rf.predict_proba(X_test)[:,1]
# roc_rf = roc_auc_score(y_test, y_prob_rf)
# pr_rf  = average_precision_score(y_test, y_prob_rf)
# print(f"Random Forest test ROC-AUC: {roc_rf:.3f}, PR-AUC: {pr_rf:.3f}")

# # ------------------------
# # LightGBM tuning via Optuna
# # ------------------------
# best_pr = 0
# best_roc = 0

# def lgb_objective(trial):
#     global best_pr, best_roc
#     params = {
#         "objective": "binary",
#         "metric": "auc",
#         "class_pos_weight": imbalance_ratio,
#         "boosting_type": "gbdt",
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "num_leaves": trial.suggest_int("num_leaves", 16, 256),
#         "max_depth": trial.suggest_int("max_depth", 3, 8),
#         "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
#         "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
#         "verbosity": -1, "seed": 42
#     }
#     dtrain = lgb.Dataset(X_train, label=y_train)
#     dvalid = lgb.Dataset(X_val, label=y_val)
#     pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
#     model = lgb.train(params, dtrain, num_boost_round=1000,
#                       valid_sets=[dvalid],
#                       callbacks=[pruning_callback, lgb.early_stopping(50, verbose=False)])
#     y_pred = model.predict(X_val, num_iteration=model.best_iteration)
#     roc = roc_auc_score(y_val, y_pred)
#     pr  = average_precision_score(y_val, y_pred)
#     trial.set_user_attr("roc_auc", roc)
#     trial.set_user_attr("pr_auc", pr)
#     if pr > best_pr:
#         best_pr = pr
#         best_roc = roc
#         print(f"New best trial {trial.number}: ROC={roc:.4f}, PR={pr:.4f}")
#     return roc

# study_lgb = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
# study_lgb.optimize(lgb_objective, n_trials=100)

# best_params_lgb = study_lgb.best_trial.params
# best_params_lgb.update({"objective":"binary","metric":"auc","verbosity":-1})
# final_lgb = lgb.LGBMClassifier(**best_params_lgb, n_estimators=1000)
# final_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])

# y_pred_lgb = final_lgb.predict_proba(X_test)[:,1]
# roc_lgb = roc_auc_score(y_test, y_pred_lgb)
# pr_lgb  = average_precision_score(y_test, y_pred_lgb)
# print(f"LightGBM test ROC-AUC: {roc_lgb:.3f}, PR-AUC: {pr_lgb:.3f}")

# # ------------------------
# # XGBoost tuning via Optuna
# # ------------------------
# best_pr = 0
# best_roc = 0

# def xgb_objective(trial):
#     global best_pr, best_roc
#     params = {
#         "objective":"binary:logistic",
#         "eval_metric":"auc",
#         "class_pos_weight":imbalance_ratio,
#         "tree_method":"hist",
#         "max_depth": trial.suggest_int("max_depth", 3, 8),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
#         "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
#         "random_state": 42
#     }
#     dtrain = xgb.DMatrix(X_train, label=y_train)
#     dvalid = xgb.DMatrix(X_val, label=y_val)
#     pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-auc")
#     model = xgb.train(params, dtrain, num_boost_round=1000,
#                       evals=[(dvalid, "validation_0")],
#                       early_stopping_rounds=50,
#                       verbose_eval=False,
#                       callbacks=[pruning_callback])
#     y_pred = model.predict(dvalid, iteration_range=(0, model.best_iteration))
#     roc = roc_auc_score(y_val, y_pred)
#     pr  = average_precision_score(y_val, y_pred)
#     trial.set_user_attr("roc_auc", roc)
#     trial.set_user_attr("pr_auc", pr)
#     if pr > best_pr:
#         best_pr = pr
#         best_roc = roc
#         print(f"New best trial {trial.number}: ROC={roc:.4f}, PR={pr:.4f}")
#     return roc

# study_xgb = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
# study_xgb.optimize(xgb_objective, n_trials=100)

# best_params_xgb = study_xgb.best_trial.params
# best_params_xgb.update({"objective":"binary:logistic","eval_metric":"auc","tree_method":"hist"})
# final_xgb = xgb.XGBClassifier(**best_params_xgb, n_estimators=1000)
# final_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# y_pred_xgb = final_xgb.predict_proba(X_test)[:,1]
# roc_xgb = roc_auc_score(y_test, y_pred_xgb)
# pr_xgb  = average_precision_score(y_test, y_pred_xgb)
# print(f"XGBoost test ROC-AUC: {roc_xgb:.3f}, PR-AUC: {pr_xgb:.3f}")