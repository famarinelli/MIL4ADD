import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
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

def _load_splits_from_csv(csv_path: str):
    """
    Carica gli split da un file CSV.
    Il CSV ha colonne: train, val, test (e opzionalmente internal_val)
    I valori sono gli ID dei partecipanti.
    
    Args:
        csv_path (str): Percorso al file CSV contenente uno split.
    
    Returns:
        dict: Dizionario con le liste di partecipanti per train, val/internal_val, test.
    """
    df = pd.read_csv(csv_path)
    split_dict = {}
    
    for col in df.columns:
        # Carica i valori dalla colonna e rimuove i NaN
        participant_ids = df[col].dropna().tolist()
        # Converti a int se possibile
        participant_ids = [int(pid) if isinstance(pid, (int, float)) else pid for pid in participant_ids]
        split_dict[col] = participant_ids
    
    return split_dict

def _load_splits_from_directory(splits_dir: str):
    """
    Carica tutti gli split da una cartella contenente file CSV.
    Aspetta file nominati come split_0.csv, split_1.csv, ecc.
    
    Args:
        splits_dir (str): Percorso alla cartella contenente i file split_*.csv
    
    Returns:
        list: Una lista di dizionari con gli split per ogni fold.
    """
    splits = []
    split_idx = 0
    
    while True:
        split_file = os.path.join(splits_dir, f"split_{split_idx}.csv")
        if not os.path.exists(split_file):
            break
        splits.append(_load_splits_from_csv(split_file))
        split_idx += 1
    
    if splits:
        print(f"Caricati {len(splits)} fold dalla cartella: {splits_dir}")
    
    return splits

def _save_splits_to_csv(splits: list, splits_dir: str):
    """
    Salva gli split in CSV separati per ogni fold.
    Crea file nominati split_0.csv, split_1.csv, ecc. con colonne train, val, test (o internal_val).
    
    Args:
        splits (list): Lista di dizionari con gli split per ogni fold.
        splits_dir (str): Cartella dove salvare i file CSV.
    """
    os.makedirs(splits_dir, exist_ok=True)
    
    for fold_idx, split_dict in enumerate(splits):
        # Determina il numero massimo di partecipanti in una colonna
        max_len = max(len(ids) for ids in split_dict.values())
        
        # Crea un dizionario con le colonne, riempiendo con NaN
        data = {}
        for split_type, participant_ids in split_dict.items():
            # Crea una lista riempita con NaN per allineare le lunghezze
            padded_ids = participant_ids + [np.nan] * (max_len - len(participant_ids))
            data[split_type] = padded_ids
        
        df = pd.DataFrame(data)
        split_filepath = os.path.join(splits_dir, f"split_{fold_idx}.csv")
        df.to_csv(split_filepath, index=False)
        print(f"Split {fold_idx} salvato in: {split_filepath}")
    
    print(f"Totale {len(splits)} split salvati in: {splits_dir}")

def _find_splits_directory(splits_dir: str, k: int, random_state: int, internal_val_size: float = 0.0):
    """
    Trova la cartella degli split per un dato setting.
    
    Args:
        splits_dir (str): Cartella base degli split.
        k (int): Numero di fold.
        random_state (int): Seed.
        internal_val_size (float): Percentuale di validation interna.
    
    Returns:
        str: Percorso alla cartella degli split, o None se non esiste.
    """
    if internal_val_size > 0:
        internal_val_pct = int(internal_val_size * 100)
        folder_name = f"k_{k}_splits_seed_{random_state}__in_val_{internal_val_pct}"
    else:
        folder_name = f"k_{k}_splits_seed_{random_state}"
    
    folder_path = os.path.join(splits_dir, folder_name)
    
    if os.path.isdir(folder_path):
        return folder_path
    else:
        return None

def get_k_fold_splits(processed_data_path: str, k: int, splits_dir: str, random_state: int = 42, internal_val_size: float = 0.0, splits_csv_path: str = None):
    """
    Crea o carica gli split per la K-Fold Cross-Validation stratificata.
    Se internal_val_size > 0, crea split aggiuntivi per validation e test interno.
    Se splits_csv_path è fornito, carica gli split da quel file CSV.
    
    Gli split vengono organizzati in cartelle con uno split per file CSV:
    - Per ogni setting (k, seed, internal_val_size) crea una cartella
    - Dentro la cartella: split_0.csv, split_1.csv, ecc.
    - Ogni CSV ha colonne: train, val/internal_val, test
    
    Args:
        processed_data_path (str): Percorso al file CSV con i dati processati.
        k (int): Numero di fold.
        splits_dir (str): Cartella base dove salvare/caricare i file degli split.
        random_state (int): Seed per la riproducibilità.
        internal_val_size (float): Percentuale di training da usare come validation interna (0.0 = disabilitato).
        splits_csv_path (str): Percorso al CSV contenente gli split predefiniti (opzionale).

    Returns:
        list: Una lista di dizionari. Senza internal_val: {'train': [...], 'val': [...]}
              Con internal_val: {'train': [...], 'internal_val': [...], 'test': [...]}
    """
    # 0. Se splits_csv_path è fornito, carica gli split da un singolo file CSV esterno
    if splits_csv_path is not None:
        if os.path.isdir(splits_csv_path):
            # È una cartella con split_*.csv
            print(f"Caricamento degli split dalla cartella: {splits_csv_path}")
            return _load_splits_from_directory(splits_csv_path)
        else:
            raise ValueError(f"splits_csv_path deve essere una cartella con split_*.csv: {splits_csv_path}")
    
    # Crea la cartella base degli split se non esiste
    os.makedirs(splits_dir, exist_ok=True)
    
    # 1. Prova a trovare e caricare gli split da una cartella esistente
    existing_dir = _find_splits_directory(splits_dir, k, random_state, internal_val_size)
    if existing_dir is not None:
        print(f"Caricamento degli split dalla cartella: {existing_dir}")
        splits = _load_splits_from_directory(existing_dir)
        if splits:
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
        train_full_ids = X[train_indices].tolist()
        test_ids = X[val_indices].tolist()
        
        if internal_val_size > 0:
            # Se internal_val_size è specificato, splittare il training set in train/val interno
            # Ricava le label per lo split stratificato
            train_full_labels = y[train_indices].tolist()
            
            # Split stratificato
            real_train_ids, internal_val_ids = train_test_split(
                train_full_ids,
                test_size=internal_val_size,
                stratify=train_full_labels,
                random_state=random_state
            )
            
            splits.append({
                'train': real_train_ids, 
                'internal_val': internal_val_ids,
                'test': test_ids
            })
        else:
            # Modalità standard senza internal_val
            splits.append({
                'train': train_full_ids, 
                'val': test_ids
            })

    # 3. Salva gli split in CSV separati nella cartella appropriata
    if internal_val_size > 0:
        internal_val_pct = int(internal_val_size * 100)
        folder_name = f"k_{k}_splits_seed_{random_state}__in_val_{internal_val_pct}"
    else:
        folder_name = f"k_{k}_splits_seed_{random_state}"
    
    splits_folder = os.path.join(splits_dir, folder_name)
    _save_splits_to_csv(splits, splits_folder)
    
    return splits