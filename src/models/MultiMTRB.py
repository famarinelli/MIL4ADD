import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiMTRB(nn.Module):
    def __init__(self, input_dim, num_branches=2, pooling_type='attention', hidden_dim=256, dropout=0.3, alpha=0.8, beta=5, **kwargs):
        """
        Implementazione Multi-Branch MIL parametrica.
        Accetta un numero arbitrario di tipi di embedding concatenati.
        
        Args:
            input_dim (int): Dimensione totale delle feature concatenate.
            num_branches (int): Numero di modelli/modalità (es. 2 per RoBERTa+MT5).
            alpha (float): Soglia confidenza istanza.
            beta (int): Soglia conteggio istanze positive.
        """
        super().__init__()
        
        self.num_branches = num_branches
        self.alpha = alpha
        self.beta = beta
        
        # Controllo validità dimensioni
        if input_dim % num_branches != 0:
            raise ValueError(
                f"Input dim {input_dim} non è divisibile per num_branches {num_branches}. "
                "Assicurati che tutti i modelli di embedding abbiano la stessa dimensione."
            )
        
        self.branch_dim = input_dim // num_branches
        
        # Usiamo ModuleList per creare liste di layer che PyTorch riconosce come parametri
        self.classifiers = nn.ModuleList()
        self.att_nets = nn.ModuleList()
        
        for _ in range(num_branches):
            # 1. Classificatore di istanza per questo branch
            clf = nn.Sequential(
                nn.Linear(self.branch_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
            self.classifiers.append(clf)
            
            # 2. Attention Net specifica per questo branch
            att = nn.Sequential(
                nn.Linear(self.branch_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            self.att_nets.append(att)

    def forward(self, x, mask=None):
        """
        Args:
            x: (Batch, Max_Instances, Total_Dim)
        """
        # 1. Split Input in N parti lungo l'ultima dimensione
        # x_splits è una tupla di N tensori, ognuno di dim (B, N, branch_dim)
        x_splits = torch.chunk(x, self.num_branches, dim=-1)
        
        bag_logits_list = []
        inst_logits_list = []
        
        # 2. Iteriamo su ogni branch
        for i in range(self.num_branches):
            feat = x_splits[i]
            
            # A. Instance Logits
            inst_logits = self.classifiers[i](feat) # (B, N, 1)
            inst_logits_list.append(inst_logits)
            
            # B. Attention Weights
            att_logits = self.att_nets[i](feat) # (B, N, 1)
            if mask is not None:
                att_logits = att_logits.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            att_weights = F.softmax(att_logits, dim=1)
            
            # C. Bag Logit (Somma pesata)
            bag_logit = torch.sum(inst_logits * att_weights, dim=1) # (B, 1)
            bag_logits_list.append(bag_logit)

        # 3. Fusion
        
        # A. Training: Media dei Bag Logits dei vari branch
        # Stackiamo la lista in un tensore (Num_Branches, B, 1) e facciamo media su dim 0
        all_bag_logits = torch.stack(bag_logits_list, dim=0)
        final_bag_logit = torch.mean(all_bag_logits, dim=0) # (B, 1)
        
        # B. Inference: Media dei Logits delle Istanze
        all_inst_logits = torch.stack(inst_logits_list, dim=0)
        avg_inst_logits = torch.mean(all_inst_logits, dim=0) # (B, N, 1)
        
        # Sigmoide per ottenere score 0-1
        instance_scores = torch.sigmoid(avg_inst_logits).squeeze(-1) # (B, N)
        
        if mask is not None:
            instance_scores = instance_scores * mask
        
        return final_bag_logit, instance_scores

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