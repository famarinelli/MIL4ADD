import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    RobertaModel, 
    MT5EncoderModel
)
from tqdm import tqdm

# --- Dataset Class ---
class TextDataset(Dataset):
    def __init__(self, texts, participant_ids):
        self.texts = texts
        self.participant_ids = participant_ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return str(self.texts[idx]), self.participant_ids[idx]

# --- Embedding Extraction Logic ---
def get_embeddings(model, tokenizer, loader, device, model_type):
    model.eval()
    all_embeddings = []
    all_ids = []

    print(f"Start inference with {model_type} on {device}...")
    
    with torch.no_grad():
        for texts, p_ids in tqdm(loader, desc="Generating Embeddings"):
            # Tokenizzazione
            inputs = tokenizer(
                list(texts), 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(device)

            # Forward pass
            outputs = model(**inputs)

            if model_type == 'roberta':
                # Per RoBERTa, usiamo il token [CLS] (posizione 0) come rappresentazione della frase
                # last_hidden_state shape: (batch, seq_len, hidden_size)
                embeddings = outputs.last_hidden_state[:, 0, :]
            
            elif model_type == 'mt5':
                # Per MT5, non c'è un token [CLS] esplicito pre-addestrato per classificazione.
                # La pratica comune è fare il Mean Pooling sugli stati nascosti, ignorando il padding.
                attention_mask = inputs['attention_mask']
                last_hidden_state = outputs.last_hidden_state
                
                # Espandiamo la maschera per matchare le dimensioni dell'embedding
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                
                # Somma degli stati nascosti validi
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                
                # Somma dei token validi (clamp per evitare divisione per zero)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                embeddings = sum_embeddings / sum_mask

            # Spostiamo su CPU per risparmiare memoria GPU e aggiungiamo alla lista
            all_embeddings.append(embeddings.cpu())
            all_ids.extend(p_ids.tolist()) # p_ids è un tensor di interi

    # Concateniamo tutti i batch
    final_embeddings = torch.cat(all_embeddings, dim=0)
    return final_embeddings, all_ids

def main():
    parser = argparse.ArgumentParser(description="Generate Embeddings for MIL")
    parser.add_argument("--input_file", type=str, required=True, help="Path to processed CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Dir to save embeddings")
    parser.add_argument("--model_name", type=str, required=True, help="HF Model name (e.g. roberta-base, google/mt5-small)")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()

    # 1. Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determina il tipo di modello per la logica di estrazione
    if 'roberta' in args.model_name.lower():
        model_type = 'roberta'
        print(f"Loading RoBERTa model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = RobertaModel.from_pretrained(args.model_name)
    elif 'mt5' in args.model_name.lower():
        model_type = 'mt5'
        print(f"Loading MT5 Encoder model: {args.model_name}")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        
        # 2. Usa use_safetensors=True per risolvere l'errore di sicurezza SENZA aggiornare PyTorch
        try:
            model = MT5EncoderModel.from_pretrained(args.model_name, use_safetensors=True)
            print("Modello caricato correttamente usando safetensors.")
        except Exception as e:
            # Questo blocco scatterebbe solo se il modello su HuggingFace non ha i pesi safetensors.
            # google/mt5-base LI HA, quindi questo codice non dovrebbe mai servire, 
            # ma lo lasciamo per completezza o se cambi modello in futuro.
            print(f"Errore caricamento safetensors: {e}")
            print("Tentativo fallback (richiederà PyTorch aggiornato)...")
            model = MT5EncoderModel.from_pretrained(args.model_name)
    else:
        raise ValueError("Model name must contain 'roberta' or 'mt5'")

    model.to(args.device)

    # 2. Load Data
    print(f"Loading data from {args.input_file}...")
    df = pd.read_csv(args.input_file)
    
    # Creiamo Dataset e DataLoader
    dataset = TextDataset(df['instance_text'].tolist(), df['participant_id'].tolist())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 3. Generate Embeddings
    embeddings_tensor, ids_list = get_embeddings(model, tokenizer, loader, args.device, model_type)

    # 4. Group by Participant ID (Bag creation)
    print("Grouping embeddings by Participant ID...")
    embeddings_dict = {}
    
    # Creiamo un DataFrame temporaneo per facilitare il raggruppamento
    # Nota: Manteniamo l'ordine originale delle istanze, che è cronologico
    temp_df = pd.DataFrame({'id': ids_list})
    
    # Iteriamo sugli ID unici
    unique_ids = temp_df['id'].unique()
    
    for uid in tqdm(unique_ids, desc="Grouping"):
        # Troviamo gli indici corrispondenti a questo utente
        indices = temp_df.index[temp_df['id'] == uid].tolist()
        
        # Estraiamo le righe corrispondenti dal tensore gigante
        # Questo preserva l'ordine cronologico perché indices è ordinato
        user_embeddings = embeddings_tensor[indices]
        
        embeddings_dict[str(uid)] = user_embeddings.clone()

    # 5. Save
    # Puliamo il nome del modello per il file (es. google/mt5-base -> google_mt5-base)
    safe_model_name = args.model_name.replace('/', '_')
    output_filename = f"{safe_model_name}_embeddings.pt"
    output_path = os.path.join(args.output_dir, output_filename)
    
    print(f"Saving embeddings to {output_path}...")
    torch.save(embeddings_dict, output_path)
    print("Done!")

if __name__ == "__main__":
    main()