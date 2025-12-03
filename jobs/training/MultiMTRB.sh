#!/bin/bash
#SBATCH --job-name=mil4add
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=H2020DeciderFicarra

# Attiva environment
source $SLURM_SUBMIT_DIR/venv/bin/activate

# Configurazione WandB
export WANDB_PROJECT="MIL4ADD"
# export WANDB_MODE="offline" 

# Definizione Variabili
# Nota: MultiMTRB richiede ENTRAMBI gli embedding
EMBEDDINGS="data/embeddings/roberta-base_embeddings.pt data/embeddings/google_mt5-base_embeddings.pt"
CSV_PATH="data/processed/processed_instances.csv"

# Parametri del Paper (o vicini ad essi)
MODEL="MultiMTRB"
ALPHA=0.8
BETA=5
MAX_INST=40  # Top n frasi più lunghe
EXP_NAME="MultiMTRB_Alpha${ALPHA}_Beta${BETA}_Top${MAX_INST}"

echo "Starting Training: $EXP_NAME"

python train.py \
    --processed_csv $CSV_PATH \
    --embedding_files $EMBEDDINGS \
    --k_folds 5 \
    --batch_size 128 \
    --epochs 100 \
    --lr 0.001 \
    --patience 15 \
    --model_name $MODEL \
    --alpha $ALPHA \
    --beta $BETA \
    --max_instances $MAX_INST \
    --hidden_dim 16 \
    --dropout 0.3 \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity "multimodal_decider" \
    --exp_name $EXP_NAME \
    --internal_val_size 0.15

echo "Training Finished."