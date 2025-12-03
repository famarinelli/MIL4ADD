import torch
import torch.nn as nn
from src.pooling import MeanPooling, MaxPooling, AttentionPooling

class MILModel(nn.Module):
    def __init__(self, input_dim, pooling_type='attention', hidden_dim=256, dropout=0.3):
        """
        Args:
            input_dim (int): Dimensione delle feature in input (es. 768 o 1536).
            pooling_type (str): 'mean', 'max', o 'attention'.
            hidden_dim (int): Dimensione dello strato nascosto interno.
            dropout (float): Probabilità di dropout.
        """
        super().__init__()
        self.pooling_type = pooling_type.lower()
        
        # 1. Instance Projector (Opzionale ma consigliato)
        # Trasforma l'embedding grezzo (es. RoBERTa) in uno spazio latente specifico per il task
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Pooling Layer
        if self.pooling_type == 'mean':
            self.pooling = MeanPooling()
        elif self.pooling_type == 'max':
            self.pooling = MaxPooling()
        elif self.pooling_type == 'attention':
            # L'attention lavora sullo spazio proiettato (hidden_dim)
            self.pooling = AttentionPooling(input_dim=hidden_dim, hidden_dim=hidden_dim // 2)
        else:
            raise ValueError(f"Pooling type '{pooling_type}' not supported.")
        
        # 3. Bag Classifier
        # Prende il vettore aggregato e decide: Depresso (1) o No (0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1) # Output logit (senza sigmoide, useremo BCEWithLogitsLoss)
        )

    def forward(self, x, mask=None):
        # x: (Batch, Max_Instances, Input_Dim)
        # mask: (Batch, Max_Instances)
        
        # 1. Proiezione delle istanze
        # Applichiamo il layer lineare a ogni istanza indipendentemente
        h = self.projector(x) # (B, N, Hidden_Dim)
        
        # 2. Aggregazione (Pooling)
        if self.pooling_type == 'attention':
            bag_embedding, attn_weights = self.pooling(h, mask)
        else:
            bag_embedding = self.pooling(h, mask)
            attn_weights = None # Mean/Max non producono pesi espliciti per istanza
            
        # 3. Classificazione finale
        logits = self.classifier(bag_embedding) # (B, 1)
        
        return logits, attn_weights