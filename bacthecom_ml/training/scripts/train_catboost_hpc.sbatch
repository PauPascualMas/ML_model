#!/bin/sh
#SBATCH --job-name=bact_train
#SBATCH --array=0-4
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --partition=middle_idx
#SBATCH --chdir /scratch/bi/TESTS/bacthecom_ml/training
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ===========
# Log directories
# ===========
mkdir -p scripts/logs

# ===========
# Run Optuna 
# ===========

SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:=0}
SLURM_JOB_ID=${SLURM_JOB_ID:=0}

# CatBoost
singularity exec \
    --containall \
    --bind /scratch/bi/TESTS/bacthecom_ml/training:/mnt \
    bacthecom.sif \
    python3 -u /mnt/scripts/train_catboost_optuna_hpc.py \
        --data /mnt/data/preprocessed_db.csv \
        --target mortalidad_30_dias \
        --study_name bact_ctb_mort30 \
        --model_dir /mnt/model/ \
        --results_dir /mnt/results/training/catboost \
        --storage sqlite:////mnt/model/bacthecom_catboost_optuna.db \
        --n_trials 300 \
        --seed 99 \
    > scripts/logs/catboost_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1


# XGBoost
# singularity exec \
#     --containall \
#     --bind /scratch/bi/TESTS/bacthecom_ml/training:/mnt \
#     bacthecom.sif \
#     python3 -u /mnt/scripts/train_xgboost_optuna.py \
#         --data /mnt/data/preprocessed_db.csv \
#         --target mortalidad_30_dias \
#         --study_name bact_xgb_mort30 \
#         --model_dir /mnt/model/ \
#         --results_dir /mnt/results/training/xgboost \
#         --storage sqlite:////mnt/model/bacthecom_xgboost_optuna.db \
#         --n_trials 100 \
#         --seed 42 \
#     > scripts/logs/xgboost_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1

# # LightGBM
# singularity exec \
#     --containall \
#     --bind /scratch/bi/TESTS/bacthecom_ml/training:/mnt \
#     bacthecom.sif \
#     python3 -u /mnt/scripts/train_lgbm_optuna.py \
#         --data /mnt/data/preprocessed_db.csv \
#         --target mortalidad_30_dias \
#         --study_name bact_lgbm_mort30 \
#         --model_dir /mnt/model/ \
#         --results_dir /mnt/results/training/lgbm \
#         --storage sqlite:////mnt/model/bacthecom_lgbm_optuna.db \
#         --n_trials 100 \
#         --seed 42 \
#     > scripts/logs/lgbm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1
