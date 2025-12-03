import os
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Imposta il seed per la riproducibilità degli esperimenti.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Queste impostazioni sono necessarie per la piena riproducibilità su GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_k_fold_splits(processed_data_path: str, k: int, splits_dir: str, random_state: int = 42):
    """
    Crea o carica gli split per la K-Fold Cross-Validation stratificata.
    
    Args:
        processed_data_path (str): Percorso al file CSV con i dati processati.
        k (int): Numero di fold.
        splits_dir (str): Cartella dove salvare/caricare i file degli split.
        random_state (int): Seed per la riproducibilità.

    Returns:
        list: Una lista di dizionari, dove ogni dizionario contiene gli ID di train e validation per un fold.
              Esempio: [{'train': [id1, id2, ...], 'val': [id3, id4, ...]}, ...]
    """
    # Crea la cartella degli split se non esiste
    os.makedirs(splits_dir, exist_ok=True)
    split_filename = f"k_{k}_splits_seed_{random_state}.json"
    split_filepath = os.path.join(splits_dir, split_filename)

    # 1. Controlla se il file degli split esiste già
    if os.path.exists(split_filepath):
        print(f"Caricamento degli split esistenti da: {split_filepath}")
        with open(split_filepath, 'r') as f:
            splits = json.load(f)
        return splits

    # 2. Se non esiste, crea nuovi split
    print(f"Creazione di nuovi split per k={k} con seed={random_state}...")
    df = pd.read_csv(processed_data_path)

    # Ottieni un DataFrame con ID unici dei partecipanti e le loro etichette
    participants_df = df[['participant_id', 'bag_label']].drop_duplicates().reset_index(drop=True)
    
    # Prepara i dati per StratifiedKFold
    X = participants_df['participant_id'].to_numpy()
    y = participants_df['bag_label'].to_numpy()

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    
    splits = []
    for train_indices, val_indices in skf.split(X, y):
        # Mappa gli indici numerici agli ID effettivi dei partecipanti
        train_ids = X[train_indices].tolist()
        val_ids = X[val_indices].tolist()
        
        splits.append({'train': train_ids, 'val': val_ids})

    # 3. Salva gli split in un file JSON per usi futuri
    with open(split_filepath, 'w') as f:
        json.dump(splits, f, indent=4)
    
    print(f"Split salvati in: {split_filepath}")
    return splits