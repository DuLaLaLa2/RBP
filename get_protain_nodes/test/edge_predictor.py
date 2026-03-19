import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgePredictor(nn.Module):
    """
    边预测器：学习节点之间是否存在边
    使用双线性变换预测边概率
    """
    def __init__(self, node_feature_dim, hidden_dim=256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.edge_predictor = nn.Bilinear(hidden_dim, hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Bilinear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x):
        return self.encoder(x)
    
    def predict_edge(self, emb_i, emb_j):
        return self.edge_predictor(emb_i, emb_j)
    
    def forward(self, x_i, x_j):
        emb_i = self.encode(x_i)
        emb_j = self.encode(x_j)
        logits = self.predict_edge(emb_i, emb_j)
        return logits.squeeze(-1)