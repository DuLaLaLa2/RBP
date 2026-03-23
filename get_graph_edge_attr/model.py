import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# === 自带边特征的残差GAT层 ===
class ResidualGATLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.6, edge_dim=4):
        super().__init__()
        self.gat = GATConv(
            hidden_dim, hidden_dim, heads=1, concat=False,
            dropout=dropout, edge_dim=edge_dim
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        residual = x
        out = self.gat(x, edge_index, edge_attr)
        out = F.elu(out)
        out = torch.cat([out, residual], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


# === 深度残差GAT（支持边特征） ===
class DeepResidualGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=4,
                 heads=4, dropout=0.6, edge_dim=4):
        super().__init__()

        # 输入层（多头注意力 + 降维）
        self.input_gat = GATConv(
            in_dim, hidden_dim, heads=heads, concat=True, edge_dim=edge_dim
        )
        self.input_proj = nn.Linear(hidden_dim * heads, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # 隐藏层（残差堆叠）
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(
                ResidualGATLayer(hidden_dim, dropout, edge_dim=edge_dim)
            )

        # 输出层
        self.output_gat = GATConv(
            hidden_dim, hidden_dim, heads=1, concat=False, edge_dim=edge_dim
        )
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        # 输入层
        x = self.input_gat(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.dropout(x)

        # 残差层
        for layer in self.hidden_layers:
            x = layer(x, edge_index, edge_attr)

        # 输出层
        x = F.elu(self.output_gat(x, edge_index, edge_attr))
        x = self.dropout(x)
        out = self.output_proj(x)

        return F.log_softmax(out, dim=1)


# === Focal Loss （解决样本不平衡） ===
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.nll_loss(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
