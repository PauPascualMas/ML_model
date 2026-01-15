#!/bin/sh
#SBATCH --job-name=bact_train
#SBATCH --array=0-7          
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=middle_idx
#SBATCH --output=logs/train_%A_%a.log
#SBATCH --error=logs/train_%A_%a.err

# ===============================
# Paths
# ===============================
DATA=/data/ucct/bi/scratch_tmp/bi/TESTS/bacthecom/training/data/preprocessed_df.csv
STORAGE=sqlite:////data/ucct/bi/scratch_tmp/bi/TESTS/bacthecom/model/bacthecom_catboost.db
STUDY=bact_ctb_mort30

# ===============================
# Run Optuna (ARRAY)
# ===============================
python catboost_optuna_array.py \
  --data ${DATA} \
  --target mortalidad_30_dias \
  --study_name ${STUDY} \
  --storage ${STORAGE} \
  --n_trials 100 \
  --seed 42
