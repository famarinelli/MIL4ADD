import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class MILDataset(Dataset):
    def __init__(self, embedding_paths, labels_path, participant_ids=None, max_instances=None):
        """
        Args:
            embedding_paths (list[str]): Lista di percorsi ai file .pt contenenti gli embedding.
            labels_path (str): Percorso al file 'processed_instances.csv' che contiene testo, ID e label.
            participant_ids (list[int], optional): Lista di ID da includere. Se None, usa tutti.
            max_instances (int, optional): Se specificato, tiene solo le Top N istanze più lunghe per paziente.
        """
        self.embedding_paths = embedding_paths
        self.max_instances = max_instances
        
        # 1. Carichiamo il DataFrame unico (Dati + Label)
        # labels_path ora punta a data/processed/processed_instances.csv
        df = pd.read_csv(labels_path)
        
        # Creiamo la mappa delle label: {participant_id: label}
        self.label_map = df.groupby('participant_id')['bag_label'].first().to_dict()

        # 2. Calcolo lunghezze (Solo se serve il filtro)
        if self.max_instances is not None:
            # Calcoliamo il numero di parole (approssimato)
            df['word_count'] = df['instance_text'].apply(lambda x: len(str(x).split()))
        
        # 3. Caricamento degli Embedding
        self.embeddings_dicts = []
        for path in embedding_paths:
            print(f"Caricamento embedding da: {path}")
            data = torch.load(path, map_location='cpu') 
            self.embeddings_dicts.append(data)
        
        # 4. Definizione degli ID da usare
        if participant_ids is None:
            self.participant_ids = list(self.embeddings_dicts[0].keys())
        else:
            self.participant_ids = [str(pid) for pid in participant_ids]
            
        self._validate_ids()

        # 5. Pre-calcolo degli indici da selezionare (Top N Longest)
        self.indices_map = {}
        
        if self.max_instances is not None:
            print(f"Applicazione filtro: Top {self.max_instances} istanze più lunghe per bag.")
            
            # Raggruppiamo per partecipante
            grouped = df.groupby('participant_id')
            
            for pid in self.participant_ids:
                try:
                    # Estraiamo le righe mantenendo l'ordine originale
                    user_rows = grouped.get_group(int(pid))
                    
                    # Resettiamo l'indice per allinearlo con il tensore (0 a N_istanze)
                    user_rows = user_rows.reset_index(drop=True)
                    
                    # Ordiniamo per lunghezza e prendiamo i top N
                    top_rows = user_rows.sort_values(by='word_count', ascending=False).head(self.max_instances)
                    
                    # Salviamo gli indici relativi
                    self.indices_map[str(pid)] = top_rows.index.tolist()
                    
                except KeyError:
                    # Caso raro: ID presente negli embedding ma non nel CSV (non dovrebbe succedere se i file sono allineati)
                    print(f"Warning: ID {pid} non trovato nel CSV ma presente negli embedding.")
                    self.indices_map[str(pid)] = None
        else:
            # Nessun filtro: prendiamo tutto
            self.indices_map = {str(pid): None for pid in self.participant_ids}


    def _validate_ids(self):
        """
        Controlla che tutti gli ID esistano in tutti i dizionari di embedding e abbiano label.
        Rimuove gli ID problematici dal dataset (anzichè sollevare eccezioni).
        """
        valid_ids = []
        removed_count = 0
        
        for pid in self.participant_ids:
            is_valid = True
            
            # Controlla se l'ID esiste in tutti i dizionari di embedding
            for i, emb_dict in enumerate(self.embeddings_dicts):
                if pid not in emb_dict:
                    is_valid = False
                    break
            
            # Controlla se l'ID ha una label associata
            if is_valid and int(pid) not in self.label_map:
                is_valid = False
            
            if is_valid:
                valid_ids.append(pid)
            else:
                removed_count += 1
        
        # Aggiorna la lista di participant_ids
        self.participant_ids = valid_ids
        
        # Stampa messaggio se sono stati rimossi pazienti
        if removed_count > 0:
            print(f"⚠️  {removed_count} pazienti rimossi dal dataset (non trovati in embedding o label)")


    def __len__(self):
        return len(self.participant_ids)

    def __getitem__(self, idx):
        pid = self.participant_ids[idx]

        # Indici da selezionare per questo utente
        indices_to_keep = self.indices_map[pid]
        
        # 1. Recupera e Concatena le Features
        features_list = []
        for emb_dict in self.embeddings_dicts:
            full_tensor = emb_dict[pid] # (Tot_Istanze, Dim)
            
            if indices_to_keep is not None:
                # Selezioniamo solo le righe corrispondenti alle frasi più lunghe
                # indices_to_keep è una lista di interi [0, 5, 2...]
                selected_tensor = full_tensor[indices_to_keep]
            else:
                selected_tensor = full_tensor
                
            features_list.append(selected_tensor)
        
        # IPOTESI: Se usiamo più modelli, concateniamo lungo l'ultima dimensione (dim=1).
        # Es: RoBERTa (N, 768) + MT5 (N, 768) -> (N, 1536)
        # Controllo: Il numero di istanze (N) deve essere identico.
        if len(features_list) > 1:
            base_len = features_list[0].shape[0]
            for i, f in enumerate(features_list[1:]):
                if f.shape[0] != base_len:
                    raise ValueError(f"Disallineamento numero istanze per ID {pid}: Modello 0 ha {base_len}, Modello {i+1} ha {f.shape[0]}")
            
            features = torch.cat(features_list, dim=1)
        else:
            features = features_list[0]

        # 2. Recupera la Label
        label = self.label_map[int(pid)]
        
        # Restituiamo un dizionario
        return {
            "id": pid,
            "features": features,      # Tensor (N_istanze, Total_Dim)
            "label": torch.tensor(label, dtype=torch.float32) # Float per la Binary Cross Entropy
        }

def mil_collate_fn(batch):
    """
    Gestisce batch di bag con numero variabile di istanze.
    Effettua il padding delle features.
    """
    # batch è una lista di dizionari restituiti da __getitem__
    
    ids = [item['id'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    features_list = [item['features'] for item in batch]
    
    # Calcoliamo le lunghezze per il padding
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths)
    feature_dim = features_list[0].shape[1]
    batch_size = len(batch)
    
    # Creiamo il tensore padded (Batch, Max_Len, Feature_Dim) inizializzato a 0
    padded_features = torch.zeros(batch_size, max_len, feature_dim)
    
    # Creiamo la maschera (Batch, Max_Len) inizializzata a 0 (False)
    # 1 indica un'istanza reale, 0 indica padding
    mask = torch.zeros(batch_size, max_len)
    
    for i, (feat, length) in enumerate(zip(features_list, lengths)):
        padded_features[i, :length, :] = feat
        mask[i, :length] = 1
        
    return {
        "ids": ids,
        "features": padded_features, # (B, Max_N, D)
        "mask": mask,                # (B, Max_N)
        "labels": labels,            # (B)
        "lengths": lengths           # List[int]
    }