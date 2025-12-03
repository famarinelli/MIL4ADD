#!/bin/bash
#SBATCH --job-name=mil4add
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=H2020DeciderFicarra

# Attiva environment
source $SLURM_SUBMIT_DIR/venv/bin/activate

# Configurazione WandB (assicurati di aver fatto wandb login prima)
export WANDB_PROJECT="MIL4ADD"
# export WANDB_MODE="offline" # Decommenta se il nodo non ha internet

# Definizione Variabili
EXP_NAME="MultiMTRB_Attention"
EMBEDDINGS="data/embeddings/roberta-base_embeddings.pt data/embeddings/google_mt5-base_embeddings.pt"

echo "Starting Training: $EXP_NAME"

python train.py \
    --processed_csv "data/processed/processed_instances.csv" \
    --embedding_files $EMBEDDINGS \
    --k_folds 5 \
    --batch_size 8 \
    --epochs 50 \
    --lr 0.00005 \
    --patience 10 \
    --model_name "MILModel" \
    --pooling "attention" \
    --hidden_dim 64 \
    --dropout 0.3 \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity "multimodal_decider" \
    --exp_name $EXP_NAME

echo "Training Finished."