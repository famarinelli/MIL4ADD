import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        # x: (Batch, Max_Instances, Hidden_Dim)
        # mask: (Batch, Max_Instances) - 1 per dati reali, 0 per padding
        
        if mask is None:
            return torch.mean(x, dim=1)
        
        # Espandiamo la maschera per matchare le dimensioni delle feature
        # mask: (B, N) -> (B, N, 1)
        mask_expanded = mask.unsqueeze(-1)
        
        # Sommiamo solo le istanze valide
        sum_embeddings = torch.sum(x * mask_expanded, dim=1)
        
        # Contiamo quante istanze valide ci sono (evitando divisione per zero)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        return sum_embeddings / sum_mask

class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        # x: (Batch, Max_Instances, Hidden_Dim)
        
        if mask is not None:
            # Settiamo il padding a un valore molto basso (-inf) così non viene mai scelto dal max
            mask_expanded = mask.unsqueeze(-1)
            x = x.masked_fill(mask_expanded == 0, -1e9)
            
        # Max pooling sulla dimensione delle istanze (dim=1)
        max_emb, _ = torch.max(x, dim=1)
        return max_emb

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=64): # hidden_dim ridotto di default
        super().__init__()
        # Semplifichiamo: Solo una proiezione lineare + Tanh
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask=None):
        # x: (Batch, N, Input_Dim)
        
        # Calcolo score: (B, N, 1)
        scores = self.attention(x) 
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=1) 
        weighted_sum = torch.sum(x * attn_weights, dim=1)
        
        return weighted_sum, attn_weights