# MIL4ADD: Multi-Instance Learning for Automatic Depression Detection

**MIL4ADD** is a modular deep learning framework designed to detect depression from clinical doctor-patient interview transcripts. This project replicates and extends the methodology outlined in the paper: **"Optimizing depression detection in clinical doctor-patient interviews using a multi-instance learning framework"** (Zhang et al., 2025).

## 📌 Project Overview
Clinical interviews are often long and contain a mix of depressive cues and "normal" conversational filler. Traditional NLP methods that treat an entire transcript as a single document often suffer from information loss or noise. 

This project implements a **Multiple Instance Learning (MIL)** framework where:
- A **Bag** represents an entire interview (a patient).
- **Instances** are individual Question-Answer (Q&A) pairs within that interview.
- The model aggregates instance features to produce a single prediction per patient.

The core of this repository is the **MultiMTRB** model, the ensemble architecture proposed in the paper. It fuses features from **RoBERTa** and **MT5** using instance-level classification and a custom threshold-based decision logic ($\alpha$ and $\beta$ hyperparameters) to improve both interpretability and detection accuracy.

---

## 🛠 Features
- **Modular Model Support**: Dynamic loading of models from `src/models/`.
- **Multi-Branch Ensemble**: Parametric support for multiple embedding types (RoBERTa, MT5, etc.).
- **Paper-Faithful Logic**: Implementation of $\alpha/\beta$ instance-counting rules and Monte Carlo (MC) Dropout.
- **Robust Training**: Stratified K-Fold cross-validation with internal validation splits and automated Early Stopping.
- **HPC Ready**: Integrated Slurm scripts for large-scale grid searches and embedding generation.
- **WandB Integration**: Full logging of metrics, learning rate schedules, and fold-wise performance.

---

## 📂 Repository Structure
```text
MIL4ADD/
├── data/                   # Raw, processed data and pre-computed embeddings
├── scripts/                # Preprocessing and Feature Extraction
│   ├── 01_preprocess_data.py
│   └── 02_generate_embeddings.py
├── src/                    # Core Logic
│   ├── models/             # Modular model definitions (e.g., MultiMTRB.py)
│   ├── datasets.py         # Lazy-loading MIL Dataset
│   ├── trainer.py          # Training loop logic
│   ├── pooling.py          # MIL aggregation strategies
│   └── utils.py            # Early Stopping and K-Fold utilities
├── jobs/                   # Slurm scripts for HPC
├── grid_search/            # Automated grid search infrastructure
└── train.py                # Main orchestration script
```

---

## 🚀 Getting Started

Follow these steps to set up the environment and run the pipeline.

### Step 1: Installation & Environment Setup
This project requires Python 3.8+ and a virtual environment located in the root directory named `venv`.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/MIL4ADD.git
   cd MIL4ADD
   ```

2. **Create the Virtual Environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate and Install Dependencies:**
   - **On Linux/macOS:**
     ```bash
     source venv/bin/activate
     pip install --upgrade pip
     pip install -r requirements.txt
     ```
   - **On Windows:**
     ```bash
     .\venv\Scripts\activate
     pip install -r requirements.txt
     ```

### Step 2: Data Preprocessing
Clean the raw transcripts and aggregate Ellie (Interviewer) and Participant turns into Q&A pairs. This script is currently tailored for the **DAIC-WOZ** dataset format.
```bash
python scripts/01_preprocess_data.py
```

### Step 3: Feature Extraction (Embedding Generation)
Generate embeddings offline to decouple heavy Transformer inference from the MIL training loop.
```bash
# Generate RoBERTa embeddings
python scripts/02_generate_embeddings.py --model_name roberta-base --output_dir data/embeddings

# Generate MT5 embeddings
python scripts/02_generate_embeddings.py --model_name google/mt5-base --output_dir data/embeddings
```

### Step 4: Training
You can run a standard training session or the **MultiMTRB** ensemble.

**Example: Running MultiMTRB (Paper Configuration)**
```bash
python train.py \
    --model_name MultiMTRB \
    --embedding_files data/embeddings/roberta-base_embeddings.pt data/embeddings/google_mt5-base_embeddings.pt \
    --alpha 0.8 \
    --beta 5 \
    --max_instances 40 \
    --mc_dropout_samples 10 \
    --lr 0.001 \
    --batch_size 128 \
    --k_folds 5 \
    --exp_name paper_replication
```

---

## 🔍 Key Hyperparameters
- `--alpha`: Confidence threshold for an instance to be considered "strongly depressive".
- `--beta`: Minimum number of depressive instances (score > 0.5) required to classify a bag as positive.
- `--max_instances`: Selects the Top-N longest Q&A pairs (responses rich in content) per interview.
- `--mc_dropout_samples`: Number of forward passes during evaluation to estimate prediction uncertainty (Monte Carlo Dropout).

---

## 📊 Grid Search
To optimize performance, use the automated grid search tool. It generates combinations based on a JSON config and submits them to Slurm.

1. Configure your search space in `grid_config.json`.
2. Launch the search:
   ```bash
   python grid_search/grid_search.py
   ```

---

## 📝 Note on Dataset Support
Currently, the preprocessing scripts are tailored specifically for the **DAIC-WOZ** (Wizard-of-Oz) dataset transcripts (tab-separated speaker/value format).