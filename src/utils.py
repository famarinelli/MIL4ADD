import os
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

def _load_splits_from_csv(csv_path: str):
    """
    Carica gli split da un file CSV.
    Il CSV deve avere colonne: fold_idx, split_type (train/internal_val/val/test), participant_id
    
    Args:
        csv_path (str): Percorso al file CSV contenente gli split.
    
    Returns:
        list: Una lista di dizionari con gli split per ogni fold.
    """
    df = pd.read_csv(csv_path)
    
    # Raggruppa per fold
    splits = []
    for fold_idx in sorted(df['fold_idx'].unique()):
        fold_data = df[df['fold_idx'] == fold_idx]
        split_dict = {}
        
        for split_type in fold_data['split_type'].unique():
            participant_ids = fold_data[fold_data['split_type'] == split_type]['participant_id'].tolist()
            # Converti a int se possibile
            participant_ids = [int(pid) if isinstance(pid, (int, float)) else pid for pid in participant_ids]
            split_dict[split_type] = participant_ids
        
        splits.append(split_dict)
    
    print(f"Caricati {len(splits)} fold dal CSV: {csv_path}")
    return splits

def _save_splits_to_csv(splits: list, csv_path: str):
    """
    Salva gli split in un file CSV.
    
    Args:
        splits (list): Lista di dizionari con gli split per ogni fold.
        csv_path (str): Percorso dove salvare il file CSV.
    """
    rows = []
    for fold_idx, split_dict in enumerate(splits):
        for split_type, participant_ids in split_dict.items():
            for pid in participant_ids:
                rows.append({
                    'fold_idx': fold_idx,
                    'split_type': split_type,
                    'participant_id': pid
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Split salvati in CSV: {csv_path}")

def _find_splits_file(splits_dir: str, k: int, random_state: int, internal_val_size: float = 0.0):
    """
    Trova il file CSV degli split nella cartella.
    
    Args:
        splits_dir (str): Cartella degli split.
        k (int): Numero di fold.
        random_state (int): Seed.
        internal_val_size (float): Percentuale di validation interna.
    
    Returns:
        str: Percorso al file CSV trovato, o None se non esiste.
    """
    if internal_val_size > 0:
        internal_val_pct = int(internal_val_size * 100)
        base_name = f"k_{k}_splits_seed_{random_state}__in_val_{internal_val_pct}"
    else:
        base_name = f"k_{k}_splits_seed_{random_state}"
    
    csv_path = os.path.join(splits_dir, f"{base_name}.csv")
    
    if os.path.exists(csv_path):
        return csv_path
    else:
        return None

def get_k_fold_splits(processed_data_path: str, k: int, splits_dir: str, random_state: int = 42, internal_val_size: float = 0.0, splits_csv_path: str = None):
    """
    Crea o carica gli split per la K-Fold Cross-Validation stratificata.
    Se internal_val_size > 0, crea split aggiuntivi per validation e test interno.
    Se splits_csv_path è fornito, carica gli split da quel file CSV.
    Quando crea nuovi split, li salva in CSV.
    Quando carica split da splits_dir, cerca il file CSV.
    
    Args:
        processed_data_path (str): Percorso al file CSV con i dati processati.
        k (int): Numero di fold.
        splits_dir (str): Cartella dove salvare/caricare i file degli split.
        random_state (int): Seed per la riproducibilità.
        internal_val_size (float): Percentuale di training da usare come validation interna (0.0 = disabilitato).
        splits_csv_path (str): Percorso al CSV contenente gli split predefiniti (opzionale).

    Returns:
        list: Una lista di dizionari. Senza internal_val: {'train': [...], 'val': [...]}
              Con internal_val: {'train': [...], 'internal_val': [...], 'test': [...]}
    """
    # 0. Se splits_csv_path è fornito, carica gli split da CSV
    if splits_csv_path is not None:
        if os.path.exists(splits_csv_path):
            print(f"Caricamento degli split da CSV esterno: {splits_csv_path}")
            return _load_splits_from_csv(splits_csv_path)
        else:
            raise FileNotFoundError(f"File CSV degli split non trovato: {splits_csv_path}")
    
    # Crea la cartella degli split se non esiste
    os.makedirs(splits_dir, exist_ok=True)
    
    # 1. Prova a trovare e caricare il file CSV degli split in splits_dir
    existing_file = _find_splits_file(splits_dir, k, random_state, internal_val_size)
    if existing_file is not None:
        print(f"Caricamento degli split esistenti da: {existing_file}")
        return _load_splits_from_csv(existing_file)

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

    # 3. Salva gli split in CSV (nuovo default)
    if internal_val_size > 0:
        internal_val_pct = int(internal_val_size * 100)
        split_filename = f"k_{k}_splits_seed_{random_state}__in_val_{internal_val_pct}.csv"
    else:
        split_filename = f"k_{k}_splits_seed_{random_state}.csv"
    
    split_filepath = os.path.join(splits_dir, split_filename)
    _save_splits_to_csv(splits, split_filepath)
    
    return splits