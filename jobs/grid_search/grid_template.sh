#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00 
#SBATCH --account=H2020DeciderFicarra

# Nota: --time è basso (20 min) perché i tuoi training sono veloci. 
# Aumentalo se necessario.

# 1. Attiva Environment
source $SLURM_SUBMIT_DIR/venv/bin/activate

# 2. Setup WandB
export WANDB_PROJECT="MIL4ADD"
# export WANDB_MODE="offline"

echo "Running Grid Search Job on $(hostname)"
echo "Arguments: $PY_ARGS"

# 3. Esegui Training
# $PY_ARGS contiene tutta la stringa "--lr 0.001 --alpha 0.8 ..." costruita da python
python train.py $PY_ARGS

echo "Job Finished"