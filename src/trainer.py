import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import os
from tqdm import tqdm
from src.utils import EarlyStopping

class Trainer:
    def __init__(
        self, 
        model, 
        optimizer, 
        criterion, 
        device, 
        wandb_run,
        fold_idx,
        checkpoint_dir="outputs/checkpoints",
        mc_dropout_samples=10,
        scheduler=None 
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion # Solitamente nn.BCEWithLogitsLoss()
        self.device = device
        self.wandb_run = wandb_run
        self.fold_idx = fold_idx
        self.checkpoint_dir = checkpoint_dir
        self.scheduler = scheduler
        self.mc_dropout_samples = mc_dropout_samples 
        
        # Creiamo la cartella per i checkpoint se non esiste
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Per tenere traccia del miglior modello
        self.best_val_f1 = 0.0
        self.best_epoch = 0

    def _enable_dropout(self, model):
        """Attiva i layer di Dropout per MC Dropout."""
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Fold {self.fold_idx} - Epoch {epoch} [TRAIN]", leave=False)
        
        for batch in pbar:
            # Spostiamo i dati sul device (GPU/CPU)
            features = batch['features'].to(self.device) # (B, Max_Len, Dim)
            mask = batch['mask'].to(self.device)         # (B, Max_Len)
            labels = batch['labels'].to(self.device)     # (B)
            
            # Zero gradienti
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, _ = self.model(features, mask)
            # logits è (B, 1), labels è (B). Facciamo squeeze per avere (B)
            logits = logits.squeeze(1)
            
            # Calcolo Loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistiche
            running_loss += loss.item()
            
            # Convertiamo logits in probabilità e poi in predizioni binarie (soglia 0.5)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
            
        # Calcolo metriche di fine epoca
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        return {
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "train_f1": epoch_f1
        }

    def validate(self, val_loader, epoch):
        self.model.eval()
        
        # --- MC DROPOUT SETUP ---
        # Se richiesto, riattiviamo SOLO i layer di Dropout
        if self.mc_dropout_samples > 1:
            self._enable_dropout(self.model)
            
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_preds_custom = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Fold {self.fold_idx} - Epoch {epoch} [VAL]", leave=False)
            for batch in pbar:
                features = batch['features'].to(self.device)
                mask = batch['mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['lengths']
                
                # --- MC DROPOUT LOOP ---
                accumulated_bag_logits = None
                accumulated_instance_scores = None
                
                # Determiniamo quante iterazioni fare (1 se normale, N se MC)
                n_passes = self.mc_dropout_samples if self.mc_dropout_samples > 1 else 1
                
                for _ in range(n_passes):
                    # Forward pass
                    # Nota: MultiMTRB ritorna (bag_logit, instance_scores)
                    #       MILModel ritorna (bag_logit, attention_weights)
                    out1, out2 = self.model(features, mask)
                    
                    curr_bag_logits = out1.squeeze(1)
                    curr_second_output = out2 # Scores o Attention
                    
                    # Accumulo Bag Logits
                    if accumulated_bag_logits is None:
                        accumulated_bag_logits = curr_bag_logits
                    else:
                        accumulated_bag_logits += curr_bag_logits
                        
                    # Accumulo Second Output (Scores/Attention)
                    if accumulated_instance_scores is None:
                        accumulated_instance_scores = curr_second_output
                    else:
                        accumulated_instance_scores += curr_second_output
                
                # --- MEDIA DEI PASSAGGI ---
                avg_bag_logits = accumulated_bag_logits / n_passes
                avg_second_output = accumulated_instance_scores / n_passes
                
                # --- CALCOLO LOSS E METRICHE STANDARD ---
                # Usiamo i logit mediati
                loss = self.criterion(avg_bag_logits, labels)
                running_loss += loss.item()
                
                probs = torch.sigmoid(avg_bag_logits)
                preds = (probs > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # --- PREDIZIONE CON REGOLE ALPHA/BETA (CUSTOM) ---
                # Verifichiamo se il modello ha il metodo custom (MultiMTRB)
                if hasattr(self.model, 'predict_with_rules'):
                    # Passiamo gli score mediati (avg_second_output)
                    custom_preds = self.model.predict_with_rules(avg_second_output, mask, lengths)
                    all_preds_custom.extend(custom_preds.cpu().numpy())

        # Calcolo metriche Standard
        val_loss = running_loss / len(val_loader)
        
        output_dict = {
            "val_loss": val_loss,
            "val_acc": accuracy_score(all_labels, all_preds),
            "val_f1": f1_score(all_labels, all_preds, zero_division=0),
            "val_precision": precision_score(all_labels, all_preds, zero_division=0),
            "val_recall": recall_score(all_labels, all_preds, zero_division=0)
        } 

        # Calcolo metriche Custom (se disponibili)
        if all_preds_custom:
            # Nota: ho rimosso le virgole finali che c'erano nel tuo snippet (creavano tuple invece di float)
            output_dict["val_acc_custom"] = accuracy_score(all_labels, all_preds_custom)
            output_dict["val_f1_custom"] = f1_score(all_labels, all_preds_custom, zero_division=0)
            output_dict["val_precision_custom"] = precision_score(all_labels, all_preds_custom, zero_division=0)
            output_dict["val_recall_custom"] = recall_score(all_labels, all_preds_custom, zero_division=0)
        
        return output_dict

    def fit(self, train_loader, val_loader, test_loader=None, 
            epochs=50, patience=10, 
            monitor_metric="val_f1", monitor_mode="max"):
        """
        Esegue il training. 
        Se test_loader è fornito, valuta il test set SOLO quando il modello migliora sul val set.
        Args:
            monitor_metric (str): Nome della metrica da monitorare per early stopping (es. 'val_loss', 'val_f1').
            monitor_mode (str): 'min' (per loss) o 'max' (per score).
        """
        print(f"Starting training for Fold {self.fold_idx}...")
        

        # Inizializza Early Stopping
        early_stopping = EarlyStopping(patience=patience, mode=monitor_mode, verbose=True)
        best_val_score_display = 0.0
        
        # Teniamo traccia delle metriche del test set corrispondenti al miglior modello di validazione
        best_test_metrics = {} 
        
        for epoch in range(1, epochs + 1):
            # 1. Training Step
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 2. Validation Step (Internal Validation per Early Stopping)
            val_metrics = self.validate(val_loader, epoch)

            current_score = val_metrics[monitor_metric]

            # 3. SCHEDULER STEP (Gestione Polimorfica)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Plateau ha bisogno della metrica per sapere se ridurre
                    self.scheduler.step(current_score)
                else:
                    # Cosine e Step avanzano in base all'epoca, senza guardare la metrica
                    self.scheduler.step()
            
            # Log del Learning Rate attuale su WandB
            current_lr = self.optimizer.param_groups[0]['lr']
            self.wandb_run.log({f"fold_{self.fold_idx}/lr": current_lr, "epoch": epoch})
            
            # 4. Check Early Stopping
            early_stopping(current_score)
            
            if early_stopping.is_best:
                best_val_score_display = current_score
                
                # Salva checkpoint
                ckpt_path = os.path.join(self.checkpoint_dir, f"best_model_fold_{self.fold_idx}.pth")
                torch.save(self.model.state_dict(), ckpt_path)
                
                # Valuta Test Set (solo se siamo migliorati)
                if test_loader is not None:
                    test_metrics = self.validate(test_loader, epoch)
                    best_test_metrics = {k.replace("val_", "test_"): v for k, v in test_metrics.items()}

            # 3. Logging su WandB
            log_dict = {"epoch": epoch}
            
            # Aggiungi Train
            for k, v in train_metrics.items():
                log_dict[f"fold_{self.fold_idx}/train/{k.replace('train_', '')}"] = v
            
            # Aggiungi Val (Internal)
            for k, v in val_metrics.items():
                log_dict[f"fold_{self.fold_idx}/val/{k.replace('val_', '')}"] = v
            
            # Aggiungi Test (Solo se esiste)
            if test_loader is not None and best_test_metrics:
                for k, v in best_test_metrics.items():
                    log_dict[f"fold_{self.fold_idx}/test/{k.replace('test_', '')}"] = v

            self.wandb_run.log(log_dict)
            
            # Print console
            log_str = (f"Ep {epoch} | LR: {current_lr:.2e} | Train Loss: {train_metrics['train_loss']:.4f} | "
                       f"{monitor_metric}: {current_score:.4f}")
            
            if early_stopping.is_best:
                log_str += " [BEST]"
            
            if test_loader is not None and best_test_metrics:
                log_str += f" | Test F1 (Best Val): {best_test_metrics.get('test_f1', 0):.3f}"
            
            print(log_str)

            # Stop?
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}. Best {monitor_metric}: {best_val_score_display:.4f}")
                break
        
        # Return finale
        if test_loader is not None:
            return best_test_metrics.get('test_f1', 0.0)
        else:
            return best_val_score_display