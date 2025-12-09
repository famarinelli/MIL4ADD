import argparse
import os, datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import wandb

# Import dei nostri moduli custom
from src.utils import set_seed, get_k_fold_splits
from src.datasets import MILDataset, mil_collate_fn
from src.trainer import Trainer
from src.models import get_model_class

def main():
    # --- 1. Definizione Argomenti da Riga di Comando ---
    parser = argparse.ArgumentParser(description="MIL Training for Depression Detection")
    
    # Dati e Percorsi
    parser.add_argument("--processed_csv", type=str, default="data/processed/processed_instances.csv", help="Path al CSV processato")
    parser.add_argument("--embedding_files", nargs='+', required=True, help="Lista dei file .pt degli embedding (es. data/embeddings/roberta.pt data/embeddings/mt5.pt)")
    parser.add_argument("--splits_dir", type=str, default="splits", help="Cartella dove salvare/caricare gli split JSON")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Cartella per salvare risultati e checkpoint")
    
    # Iperparametri Training
    parser.add_argument("--k_folds", type=int, default=5, help="Numero di fold per la CV")
    parser.add_argument("--internal_val_size", type=float, default=0.15, help="Percentuale di training set da usare come validazione interna (0.0 per disabilitare)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (numero di pazienti per batch)")
    parser.add_argument("--epochs", type=int, default=50, help="Numero massimo di epoche")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "AdamW", "SGD", "RAdam"], help="Tipo di ottimizzatore")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 penalty)")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (solo per SGD)")
    parser.add_argument("--patience", type=int, default=10, help="Patience per early stopping")
    parser.add_argument("--monitor_metric", type=str, default="val_loss", help="Metrica per Early Stopping (es. val_loss, val_f1)")
    parser.add_argument("--monitor_mode", type=str, default="min", help="Mode per Early Stopping (min, max)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed per replicabilità")
    parser.add_argument("--max_instances", type=int, default=None, help="Seleziona le Top N istanze più lunghe (es. 50). Se None, usa tutte.")
    parser.add_argument("--mc_dropout_samples", type=int, default=1, help="Numero di passaggi Monte Carlo Dropout in validazione (1 = disattivato, >1 = attivo)")
    
    # --- Argomenti Scheduler ---
    parser.add_argument("--scheduler", type=str, default="Plateau", choices=["Plateau", "Cosine", "Step", "None"], help="Tipo di Scheduler")
    parser.add_argument("--lr_patience", type=int, default=5, help="Pazienza per ReduceLROnPlateau")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="Fattore di riduzione LR (per Plateau e Step)")
    parser.add_argument("--lr_step_size", type=int, default=20, help="Ogni quante epoche ridurre LR (per StepLR)")
    parser.add_argument("--lr_min", type=float, default=1e-6, help="Learning rate minimo")
    
    # Iperparametri Modello
    parser.add_argument("--model_name", type=str, required=True, choices=["MILModel", "MultiMTRB"], help="Model name")
    parser.add_argument("--alpha", type=float, default=0.8, help="Soglia confidenza istanza (solo per MultiMTRB)")
    parser.add_argument("--beta", type=int, default=5, help="Numero minimo istanze positive (solo per MultiMTRB)")
    parser.add_argument("--pooling", type=str, default="attention", choices=["mean", "max", "attention"], help="Strategia di pooling MIL")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Dimensione nascosta del modello MIL")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default="MIL-Depression", help="Nome progetto WandB")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Tuo username/team su WandB (opzionale)")
    parser.add_argument("--exp_name", type=str, default="mil_experiment", help="Nome base per l'esperimento")

    args = parser.parse_args()

    # --- 2. Setup Iniziale ---
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 4. Inizializzazione WandB Run (Spostata PRIMA della creazione cartelle) ---
    # Dobbiamo inizializzare WandB prima per avere l'ID univoco della run
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.exp_name,
        config=vars(args),
        reinit=True
    )
    
    # 2. Info K-Fold
    k_fold_dir = f"k{args.k_folds}"
    
    # 3. Timestamp e Data
    now = datetime.datetime.now()
    timestamp_str = str(round(now.timestamp()))
    date_str = now.strftime("%y_%m_%d__%H_%M")
    
    # 4. WandB Run ID (per unicità assoluta in caso di start simultanei)
    run_id = run.id
    
    # Costruzione path finale: outputs/MILModel/k5/17156234_24_05_13__10_30_runid123
    unique_run_name = f"{timestamp_str}_{date_str}_{run_id}"
    
    # Aggiorniamo args.output_dir per puntare alla cartella specifica di questa run
    run_output_dir = os.path.join(args.output_dir, args.model_name, k_fold_dir, unique_run_name)
    
    results_dir = os.path.join(run_output_dir, "results")
    checkpoints_dir = os.path.join(run_output_dir, "checkpoints")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    print(f"Output directory created at: {run_output_dir}")

    # --- 3. Caricamento Dati e Split ---
    # Otteniamo gli split (se esistono li carica, altrimenti li crea)
    splits = get_k_fold_splits(args.processed_csv, args.k_folds, args.splits_dir, args.seed)

    # Calcoliamo la dimensione dell'input dinamicamente caricando un campione
    # Questo ci evita di dover passare --input_dim a mano
    temp_ds = MILDataset(
        args.embedding_files, 
        args.processed_csv, 
        participant_ids=None,
        max_instances=args.max_instances 
    )
    input_dim = temp_ds[0]['features'].shape[1]
    print(f"Detected Input Dimension: {input_dim}")
    del temp_ds # Liberiamo memoria
    
    fold_results = []

    # --- 5. Ciclo K-Fold Cross-Validation ---
    for fold_idx, split in enumerate(splits):
        print(f"\n{'='*20} STARTING FOLD {fold_idx+1}/{args.k_folds} {'='*20}")

        # A. Preparazione Dataset
        # Split originale del K-Fold (ora interpretato come Train_Full vs Test)
        train_full_ids = split['train']
        test_ids = split['val'] # Quello che prima era val, ora è il vero Test set
        
        # Logica di Split Interno
        if args.internal_val_size > 0:
            print(f"  -> Splitting Train Full ({len(train_full_ids)}) into Train/InternalVal (size={args.internal_val_size})")
            
            # Recuperiamo le label per fare uno split stratificato
            # Creiamo un dataset temporaneo solo per leggere le label velocemente
            temp_ds_full = MILDataset(
                args.embedding_files, 
                args.processed_csv, 
                participant_ids=train_full_ids,
                max_instances=args.max_instances 
            )
            full_labels = [temp_ds_full.label_map[int(pid)] for pid in train_full_ids]
            
            # Split Stratificato
            real_train_ids, internal_val_ids = train_test_split(
                train_full_ids,
                test_size=args.internal_val_size,
                stratify=full_labels,
                random_state=args.seed
            )
            
            # Creazione Dataset
            train_ds = MILDataset(
                args.embedding_files, 
                args.processed_csv, 
                participant_ids=real_train_ids,
                max_instances=args.max_instances 
            )
            val_ds = MILDataset(
                args.embedding_files, 
                args.processed_csv, 
                participant_ids=internal_val_ids,
                max_instances=args.max_instances 
            )
            test_ds = MILDataset(
                args.embedding_files, 
                args.processed_csv, 
                participant_ids=test_ids,
                max_instances=args.max_instances 
            )
            
            # Loaders
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=mil_collate_fn)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=mil_collate_fn)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=mil_collate_fn)
            
        else:
            # Modalità Legacy (Senza Test Set separato, Val fa da Test)
            print("  -> Internal Val disabled. Using standard Train/Val split.")
            train_ds = MILDataset(
                args.embedding_files, 
                args.processed_csv, 
                participant_ids=train_full_ids,
                max_instances=args.max_instances 
            )
            val_ds = MILDataset(
                args.embedding_files, 
                args.processed_csv, 
                participant_ids=test_ids,
                max_instances=args.max_instances 
            )
            
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=mil_collate_fn)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=mil_collate_fn)
            test_loader = None # Nessun test set separato

        # --- CALCOLO AUTOMATICO POS_WEIGHT ---
        # Estraiamo tutte le label dal dataset di training corrente
        # Nota: train_ds è un MILDataset, possiamo iterare o accedere alle label se le abbiamo salvate
        # Metodo veloce: usiamo la label_map interna filtrata per gli ID di training
        train_labels = [train_ds.label_map[int(pid)] for pid in train_ds.participant_ids]
        
        num_pos = sum(train_labels)
        num_neg = len(train_labels) - num_pos
        
        # Evitiamo divisioni per zero nel caso (improbabile) di 0 positivi
        if num_pos > 0:
            weight_value = num_neg / num_pos
        else:
            weight_value = 1.0 # Fallback neutro
            
        print(f"  -> Class Imbalance: {num_neg} Neg / {int(num_pos)} Pos. Auto-calculated pos_weight: {weight_value:.4f}")
        
        pos_weight = torch.tensor([weight_value]).to(device)

        ModelClass = get_model_class(args.model_name)

        print(f"Inizializzazione modello: {args.model_name}")

        
        
        # B. Inizializzazione Modello
        if args.model_name == "MultiMTRB":
            # MultiMTRB: Richiede alpha/beta, NON usa pooling_type (ha logica interna)
            # Nota: input_dim deve essere la somma delle dim dei due modelli (es. 768+768=1536)
            model = ModelClass(
                input_dim=input_dim,
                num_branches = len(args.embedding_files),
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                alpha=args.alpha,
                beta=args.beta
            )
            
        elif args.model_name == "MILModel":
            # MILModel: Richiede pooling_type, NON usa alpha/beta
            model = ModelClass(
                input_dim=input_dim,
                pooling_type=args.pooling_type,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout
            )
            
        else:
            # Se aggiungi altri modelli in futuro, devi aggiungerli qui esplicitamente
            raise ValueError(f"Logica di inizializzazione non definita per il modello: {args.model_name}")

        # Sposta il modello sul device (GPU/CPU)
        model.to(device)
        
        # C. Ottimizzatore, Scheduler e Loss
        # --- Setup Optimizer ---
        print(f"Optimizer: {args.optimizer} | LR: {args.lr} | WD: {args.weight_decay}")
        
        if args.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=args.lr, 
                weight_decay=args.weight_decay
            )
        elif args.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=args.lr, 
                weight_decay=args.weight_decay
            )
        elif args.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(), 
                lr=args.lr, 
                momentum=args.momentum, 
                weight_decay=args.weight_decay
            )
        elif args.optimizer == "RAdam":
            optimizer = torch.optim.RAdam(
                model.parameters(), 
                lr=args.lr, 
                weight_decay=args.weight_decay
            )
        else:
            raise ValueError(f"Optimizer {args.optimizer} non supportato.")
        

        # --- Setup Scheduler ---
        scheduler = None
        if args.scheduler == "Plateau":
            print(f"Scheduler: ReduceLROnPlateau (patience={args.lr_patience}, factor={args.lr_factor})")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode=args.monitor_mode, # 'max' per F1, 'min' per Loss
                factor=args.lr_factor, 
                patience=args.lr_patience, 
                min_lr=args.lr_min,
                verbose=True
            )
        elif args.scheduler == "Cosine":
            print(f"Scheduler: CosineAnnealingLR (T_max={args.epochs})")
            # T_max è solitamente il numero totale di epoche
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=args.epochs, 
                eta_min=args.lr_min
            )
        elif args.scheduler == "Step":
            print(f"Scheduler: StepLR (step_size={args.lr_step_size}, gamma={args.lr_factor})")
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=args.lr_step_size, 
                gamma=args.lr_factor
            )
        elif args.scheduler == "None":
            print("Scheduler: Nessuno")
            scheduler = None
        
        # Setup Loss
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Adatta per classificazione binaria
        
        # D. Training
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            wandb_run=run,
            fold_idx=fold_idx,
            checkpoint_dir=checkpoints_dir,
            scheduler=scheduler,
            mc_dropout_samples=args.mc_dropout_samples 
        )
        
        best_f1 = trainer.fit(
            train_loader, 
            val_loader, 
            test_loader=test_loader, 
            epochs=args.epochs, 
            patience=args.patience,
            monitor_metric=args.monitor_metric,
            monitor_mode=args.monitor_mode 
        )
        
        # Salviamo il risultato
        fold_results.append(best_f1)
                
        print(f"Fold {fold_idx+1} completed. Best F1: {best_f1:.4f}")

    # --- 6. Aggregazione Risultati Finali ---
    mean_f1 = np.mean(fold_results)
    std_f1 = np.std(fold_results)
    
    print(f"\n{'='*20} CROSS-VALIDATION COMPLETED {'='*20}")
    print(f"F1 Scores per fold: {fold_results}")
    print(f"Average F1: {mean_f1:.4f} (+/- {std_f1:.4f})")

    # Log finale delle metriche aggregate sulla stessa run
    run.log({
        "summary/f1_mean": mean_f1,
        "summary/f1_std": std_f1,
        # Puoi aggiungere anche la lista completa se vuoi visualizzarla come tabella custom
        "summary/folds_f1": fold_results 
    })
    
    # Salvataggio risultati su file locale
    results_path = os.path.join(results_dir, f"{args.exp_name}_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Experiment: {args.exp_name}\n")
        f.write(f"Config: {vars(args)}\n")
        f.write(f"Folds F1: {fold_results}\n")
        f.write(f"Mean F1: {mean_f1:.4f}\n")
        f.write(f"Std F1: {std_f1:.4f}\n")
    
    run.finish()

if __name__ == "__main__":
    main()