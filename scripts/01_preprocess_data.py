import pandas as pd
import os
import glob
from tqdm import tqdm
import re

# --- Configurazione dei Percorsi ---
# Modifica questi percorsi se la tua struttura è diversa
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_TRANSCRIPTS_DIR = os.path.join(BASE_DIR, 'data', 'raw_transcriptions')
LABELS_PATH = os.path.join(BASE_DIR, 'data', 'labels', 'labels.csv') 
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUT_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, 'processed_instances.csv')

def clean_ellie_text(text: str) -> str:
    """
    Estrae solo il testo in linguaggio naturale dalle domande di Ellie,
    rimuovendo i tag semantici come 'okay_confirm (...)'.
    
    Args:
        text (str): Il testo originale del turno di Ellie.

    Returns:
        str: Il testo pulito, contenente solo le parti tra parentesi.
    """
    text_no_tags = re.sub(r'[a-zA-Z0-9_]+\s*\(', '(', text)
    
    # Troviamo tutte le corrispondenze tra parentesi
    matches = re.findall(r'\((.*?)\)', text_no_tags)
    
    # Se troviamo delle corrispondenze, le uniamo con uno spazio.
    if matches:
        # Pulisce ulteriormente le parentesi rimaste, nel caso di ((testo))
        cleaned_matches = [m.replace('(', '').replace(')', '') for m in matches]
        return ' '.join(cleaned_matches).strip()
    else:
        # Fallback nel caso in cui una domanda di Ellie non segua il pattern
        return text.strip()

def preprocess_transcripts():
    """
    Legge le trascrizioni grezze, le aggrega in coppie domanda-risposta (istanze)
    e le salva in un unico file CSV formattato per il framework MIL.
    """
    print("--- Inizio del Preprocessing ---")

    # 1. Carica le etichette e imposta l'ID del partecipante come indice per un accesso rapido
    try:
        labels_df = pd.read_csv(LABELS_PATH)
        # Assicuriamoci che la colonna Participant_ID sia del tipo giusto (int)
        labels_df['Participant_ID'] = labels_df['Participant_ID'].astype(int)
        labels_df.set_index('Participant_ID', inplace=True)
        print(f"Caricate {len(labels_df)} etichette da: {LABELS_PATH}")
    except FileNotFoundError:
        print(f"Errore: File delle etichette non trovato in {LABELS_PATH}")
        return

    # 2. Trova tutti i file di trascrizione
    transcript_files = glob.glob(os.path.join(RAW_TRANSCRIPTS_DIR, '*_TRANSCRIPT.csv'))
    if not transcript_files:
        print(f"Errore: Nessun file di trascrizione trovato in {RAW_TRANSCRIPTS_DIR}")
        return
    print(f"Trovati {len(transcript_files)} file di trascrizione.")

    all_instances = []

    participant_id = None
    # 3. Itera su ogni file di trascrizione
    for file_path in tqdm(transcript_files, desc="Processing Transcripts"):
        try:
            # Estrai l'ID del partecipante dal nome del file
            participant_id_str = os.path.basename(file_path).split('_')[0]
            participant_id = int(participant_id_str)

            # Ottieni l'etichetta per questo partecipante
            bag_label = labels_df.loc[participant_id, 'PHQ8_Binary']

            # Carica la trascrizione
            transcript_df = pd.read_csv(file_path, sep='\t')
            
            # Pulisci i valori nulli nella colonna 'value' che potrebbero causare errori
            transcript_df.dropna(subset=['value'], inplace=True)
            transcript_df['value'] = transcript_df['value'].astype(str)

            # --- Logica di Aggregazione delle Risposte ---
            # Identifica i "turni" di conversazione (blocchi consecutivi dello stesso speaker)
            transcript_df['turn_id'] = (transcript_df['speaker'] != transcript_df['speaker'].shift()).cumsum()
            
            # Raggruppa per turno e aggrega il testo
            turns = transcript_df.groupby('turn_id').agg(
                speaker=('speaker', 'first'),
                text=('value', lambda x: ' '.join(x))
            ).to_dict('records')

            # 4. Crea le istanze (coppie Domanda-Risposta)
            for i in range(len(turns) - 1):
                current_turn = turns[i]
                next_turn = turns[i+1]

                # Un'istanza è una domanda di Ellie seguita da una risposta del Partecipante
                if current_turn['speaker'] == 'Ellie' and next_turn['speaker'] == 'Participant':
                    # Pulisci il testo della domanda di Ellie
                    question = clean_ellie_text(current_turn['text'])
                    answer = next_turn['text'].strip()

                    # Se la domanda pulita è vuota, saltiamo l'istanza
                    # (es. un turno di Ellie che conteneva solo tag senza testo)
                    if not question:
                        continue

                    # Combiniamo domanda e risposta in un unico testo per l'istanza.
                    # Questo fornisce più contesto al modello di embedding.
                    # Usiamo un separatore chiaro.
                    instance_text = f"QUESTION: {question} [SEP] ANSWER: {answer}"
                    
                    all_instances.append({
                        'participant_id': participant_id,
                        'instance_text': instance_text,
                        'bag_label': bag_label
                    })
        except KeyError:
            print(f"Attenzione: Nessuna etichetta trovata per il partecipante {participant_id}. File saltato.")
        except Exception as e:
            print(f"Errore durante l'elaborazione del file {file_path}: {e}")

    # 5. Crea il DataFrame finale e salvalo
    if not all_instances:
        print("Nessuna istanza è stata creata. Controlla i dati di input.")
        return
        
    final_df = pd.DataFrame(all_instances)

    # Crea la cartella di output se non esiste
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"\n--- Preprocessing Completato ---")
    print(f"Creato un totale di {len(final_df)} istanze da {final_df['participant_id'].nunique()} partecipanti.")
    print(f"Dati processati salvati in: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    preprocess_transcripts()