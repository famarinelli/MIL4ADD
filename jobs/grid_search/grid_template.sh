#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00 
#SBATCH --account=H2020DeciderFicarra

source $SLURM_SUBMIT_DIR/venv/bin/activate

export WANDB_PROJECT="MIL4ADD"

echo "Running Grid Search Job on $(hostname)"

# $@ prende tutti gli argomenti passati a sbatch dopo il nome dello script
# e li passa a python. Le virgolette "$@" sono fondamentali.
python train.py "$@"

echo "Job Finished"