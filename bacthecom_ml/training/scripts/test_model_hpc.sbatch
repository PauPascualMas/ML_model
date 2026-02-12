#!/bin/sh
#SBATCH --job-name=test_plot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --partition=middle_idx
#SBATCH --output=scripts/logs/test_%A.log
#SBATCH --error=scripts/logs/test_%A.err
#SBATCH --chdir=/scratch/bi/TESTS/bacthecom_ml/training

# ===============================
# Run evaluation + plots
# ===============================
singularity exec \
    --containall \
    --bind /scratch/bi/TESTS/bacthecom_ml/training:/mnt \
    bacthecom.sif \
    python3 -u /mnt/scripts/test_model.py \
        --data /mnt/data/preprocessed_db.csv \
        --target mortalidad_30_dias \
        --study_name bact_ctb_mort30 \
        --storage sqlite:////mnt/model/bacthecom_catboost_optuna.db \
        --model_dir /mnt/model/ \
        --results_dir /mnt/results/testing/catboost \
        --seed 99
