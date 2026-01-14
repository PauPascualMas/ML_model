#!/bin/sh
#SBATCH --job-name=eval_plot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH --partition=middle_idx
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_%j.err

# ===============================
# Paths
# ===============================
DATA=/data/ucct/bi/scratch_tmp/bi/TESTS/bacthecom/training/data/preprocessed_df.csv
OUTPUT=/data/ucct/bi/scratch_tmp/bi/TESTS/bacthecom/training/results/plots
STORAGE=sqlite:////data/ucct/bi/scratch_tmp/bi/TESTS/bacthecom/model/bacthecom_catboost.db
STUDY=bact_ctb_mort30

# ===============================
# Run evaluation
# ===============================
python evaluate_and_plot.py \
  --data ${DATA} \
  --output ${OUTPUT} \
  --target mortalidad_30_dias \
  --study_name ${STUDY} \
  --storage ${STORAGE} \
  --seed 42
