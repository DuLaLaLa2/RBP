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
        # self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        residual = x
        out = self.gat(x, edge_index, edge_attr)
        out = F.elu(out)
        # out = torch.cat([out, residual], dim=-1)
        # out = self.proj(out)
        out = self.norm(out + residual)
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

        return out.squeeze(-1)   # 1维 logits，正确



# === Focal Loss （解决样本不平衡） ===
# ✅ 真正适合你任务的：二分类 Sigmoid FocalLoss
# ====================== 二分类专用 FocalLoss（完全修复版）======================
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma=5, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 论文中的 α_t，用于 y=0 的权重
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        # targets 转换为 float
        targets = targets.float()
        
        # 计算概率
        if self.logits:
            probs = torch.sigmoid(inputs)  # P_t，预测为正类的概率
        else:
            probs = inputs  # 已经是概率
        
        # 按照论文公式分段计算
        # y = 0 的情况：负类
        loss_0 = -self.alpha * (1 - probs) ** self.gamma * torch.log(1 - probs)
        
        # y = 1 的情况：正类
        loss_1 = -(1 - self.alpha) * probs ** self.gamma * torch.log(probs)
        
        # 根据真实标签选择对应的损失
        loss = targets * loss_1 + (1 - targets) * loss_0
        
        if self.reduce:
            return torch.mean(loss)
        else:
            return loss
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha  # 平衡因子，可传 list 或 None
#         self.reduction = reduction
#     def forward(self, inputs, targets):
#         # 确保标签是 float
#         targets = targets.float()
        
#         # 计算基础二分类损失
#         bce_loss = F.binary_cross_entropy_with_logits(
#             inputs, targets, reduction="none"
#         )
        
#         # 计算 Focal 核心
#         p_t = torch.exp(-bce_loss)
#         loss = (1 - p_t) ** self.gamma * bce_loss

#         # 如果有 alpha，自动处理设备 + 类型
#         if self.alpha is not None:
#             if not isinstance(self.alpha, torch.Tensor):
#                 alpha = torch.tensor(self.alpha, dtype=torch.float32, device=inputs.device)
#             else:
#                 alpha = self.alpha.to(inputs.device)
            
#             # 给正负样本加权
#             alpha_t = alpha[0] * (1 - targets) + alpha[1] * targets
#             loss = alpha_t * loss

#         if self.reduction == "mean":
#             return loss.mean()
#         return loss.sum()


