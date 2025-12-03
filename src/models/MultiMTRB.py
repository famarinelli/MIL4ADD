import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiMTRB(nn.Module):
    def __init__(self, input_dim, pooling_type='attention', hidden_dim=256, dropout=0.3, alpha=0.8, beta=5, **kwargs):
        """
        Implementazione MultiMTRB (Instance-Level MIL) con logica Alpha/Beta.
        
        Args:
            input_dim (int): Dimensione totale (MT5 + RoBERTa).
            alpha (float): Soglia di confidenza per considerare un'istanza "fortemente depressiva".
            beta (int): Numero minimo di istanze depressive (con score > 0.5) per classificare la bag come positiva.
        """
        super().__init__()
        
        if input_dim % 2 != 0:
            raise ValueError(f"Input dim {input_dim} dispari. Atteso concatenazione di 2 modelli.")
        
        self.half_dim = input_dim // 2
        self.alpha = alpha
        self.beta = beta
        
        # --- Ramo 1: RoBERTa Instance Classifier ---
        # Proietta e classifica OGNI istanza indipendentemente
        self.branch1 = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1) # Output: Logit per istanza
        )
        
        # --- Ramo 2: MT5 Instance Classifier ---
        self.branch2 = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1) # Output: Logit per istanza
        )
        
        # --- Pooling per il TRAINING (Differenziabile) ---
        # Usiamo l'Attention Pooling per aggregare i punteggi delle istanze in un punteggio bag
        # durante il training, permettendo ai gradienti di fluire.
        self.training_pooling = nn.Sequential(
            nn.Linear(self.half_dim * 2, hidden_dim), # Usa le feature originali per calcolare l'attenzione
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask=None):
        """
        Ritorna:
            bag_logit: (B, 1) - Usato per calcolare la Loss durante il training.
            instance_scores: (B, N) - Probabilità (0-1) per ogni istanza (usato per inference).
        """
        # 1. Split Input
        x1 = x[:, :, :self.half_dim]
        x2 = x[:, :, self.half_dim:]
        
        # 2. Instance-Level Classification (Logits)
        # Shape: (Batch, Max_Instances, 1)
        inst_logits1 = self.branch1(x1)
        inst_logits2 = self.branch2(x2)
        
        # 3. Fusion (Average Vote sui Logits)
        avg_inst_logits = (inst_logits1 + inst_logits2) / 2
        
        # 4. Calcolo Probabilità Istanze (Sigmoid)
        # Queste servono per la logica alpha/beta
        instance_scores = torch.sigmoid(avg_inst_logits).squeeze(-1) # (B, N)
        
        if mask is not None:
            # Azzera i punteggi del padding per non influenzare i calcoli
            instance_scores = instance_scores * mask

        # --- TRAINING PATH (Differenziabile) ---
        # Per addestrare, dobbiamo aggregare questi punteggi in un unico logit per la bag.
        # Usiamo un Attention Pooling pesato sulle feature originali.
        
        # Calcolo pesi attenzione
        att_logits = self.training_pooling(x) # (B, N, 1)
        if mask is not None:
            att_logits = att_logits.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        att_weights = F.softmax(att_logits, dim=1) # (B, N, 1)
        
        # Aggregazione pesata dei LOGIT delle istanze (MIL-Pooling approssimato)
        # Somma pesata dei logit delle istanze -> Logit della Bag
        bag_logit = torch.sum(avg_inst_logits * att_weights, dim=1) # (B, 1)
        
        return bag_logit, instance_scores

    def predict_with_rules(self, instance_scores, mask, lengths):
        """
        Applica la logica Alpha/Beta del paper (Algoritmo 1 / Fig 4) per l'Inference.
        Non differenziabile.
        
        Args:
            instance_scores: (B, N) Tensor con probabilità tra 0 e 1.
            mask: (B, N) Tensor maschera.
            lengths: List[int] lunghezze reali delle bag.
        
        Returns:
            predictions: (B) Tensor di 0 o 1.
        """
        batch_size = instance_scores.size(0)
        predictions = torch.zeros(batch_size, device=instance_scores.device)
        
        for i in range(batch_size):
            # Recupera le istanze valide per questo paziente
            valid_len = lengths[i]
            scores = instance_scores[i, :valid_len] # (N_real,)
            
            # --- LOGICA DEL PAPER (Interpretazione Algoritmo 1 & Testo pag 6) ---
            
            # 1. Conta istanze con score > Alpha
            count_above_alpha = (scores > self.alpha).sum().item()
            
            # 2. Conta istanze con score > 0.5 (Depressive Instances)
            count_depressive = (scores > 0.5).sum().item()
            
            # Criterio A: "Diagnose... if more than half of the instances exceed alpha"
            # (Nota: il testo dice "more than half", l'algoritmo è un po' vago, usiamo il testo)
            condition_a = count_above_alpha > (valid_len / 2)
            
            # Criterio B: "If there are more than Beta depressive instances (score > 0.5)"
            condition_b = count_depressive > self.beta
            
            if condition_a or condition_b:
                predictions[i] = 1.0
            else:
                predictions[i] = 0.0
                
        return predictions