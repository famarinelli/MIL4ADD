import os
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import random
import numpy as np
import torch

class EarlyStopping:
    """
    Classe helper per gestire l'Early Stopping in modo modulare.
    Supporta sia la minimizzazione (es. val_loss) che la massimizzazione (es. val_f1).
    """
    def __init__(self, patience=10, mode='max', delta=0.0, verbose=True):
        """
        Args:
            patience (int): Quante epoche aspettare dopo l'ultimo miglioramento.
            mode (str): 'min' per loss, 'max' per metriche (acc, f1).
            delta (float): Miglioramento minimo richiesto per resettare il contatore.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.is_best = False # Flag per indicare se l'epoch corrente è la migliore

        if self.mode == 'min':
            self.val_score_fn = lambda x: -x # Invertiamo il segno per usare sempre logica >
        else:
            self.val_score_fn = lambda x: x

    def __call__(self, current_score):
        score = self.val_score_fn(current_score)
        self.is_best = False

        if self.best_score is None:
            self.best_score = score
            self.is_best = True
        elif score < self.best_score + self.delta:
            # Nessun miglioramento significativo
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Miglioramento trovato
            self.best_score = score
            self.is_best = True
            self.counter = 0

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