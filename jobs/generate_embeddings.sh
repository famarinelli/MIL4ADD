#!/bin/bash
#SBATCH --job-name=gen_emb
#SBATCH --output=logs/emb_%j.out
#SBATCH --error=logs/emb_%j.err
#SBATCH --time=01:00:00          # 1 ora (aumenta per dataset completi/modelli grandi)
#SBATCH --partition=all_usr_prod # Cambia in 'gpu' o il nome della tua partizione GPU standard
#SBATCH --gres=gpu:1             # Richiede 1 GPU
#SBATCH --cpus-per-task=4        # CPU per il data loading
#SBATCH --mem=16G                # RAM di sistema
#SBATCH --account=H2020DeciderFicarra

# ROBERTA:
# sbatch jobs/generate_embeddings.sh

# MT5:
# sbatch --export=ALL,MODEL_NAME="google/mt5-base" jobs/generate_embeddings.sh

# Attiva l'ambiente virtuale
source /work/H2020DeciderFicarra/fmarinelli/mil4ADD/venv/bin/activate  # O il comando specifico per il tuo cluster (es. module load python...)

# Crea cartella logs se non esiste
mkdir -p logs

# Definisci variabili (puoi sovrascriverle lanciando sbatch --export=MODEL=...)
# Default model
# MODEL_NAME=${MODEL_NAME:-"roberta-base"} 
MODEL_NAME="google/mt5-base" # Decommenta o passa da riga di comando per MT5

INPUT_FILE="data/processed/processed_instances.csv"
OUTPUT_DIR="data/embeddings"

echo "Running embedding generation for model: $MODEL_NAME"
echo "On node: $(hostname)"

python scripts/02_generate_embeddings.py \
    --input_file $INPUT_FILE \
    --output_dir $OUTPUT_DIR \
    --model_name $MODEL_NAME \
    --batch_size 32 \
    --device cuda

echo "Job finished."