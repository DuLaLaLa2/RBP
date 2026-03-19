import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from copy import deepcopy
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef


# ===================================================== 模型定义 (与GAN-GNN.py一致) =====================================================
class GATClassifier(nn.Module):
    def __init__(self, in_dim: int = 1408, hidden_dim: int = 256, out_dim: int = 2, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        return self.classifier(x)


@torch.no_grad()
def eval_model(model, data_list, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    for data in data_list:
        data = data.to(device)
        
        # 确保 x 存在
        if data.x is None:
            data.x = torch.cat([data.struct_feat, data.seq_feat], dim=-1)
        
        logits = model(data.x, data.edge_index)
        probs = torch.softmax(logits, dim=1)[:, 1]
        
        all_preds.append(probs.cpu())
        all_labels.append(data.y.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    return all_preds, all_labels


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # ===================================================== 加载模型 =====================================================
    print("\n===== 加载模型 =====")
    checkpoint = torch.load("get_pre_result\data\wgan_312_demo5.pt", map_location=device)
    
    # 加载GAT分类器
    model = GATClassifier(in_dim=1408, hidden_dim=256, out_dim=2, heads=4).to(device)
    model.load_state_dict(checkpoint['gat'])
    print("GAT模型加载成功")
    
    # ===================================================== 加载测试数据 (117) =====================================================
    print("\n===== 加载测试数据 (117) =====")
    test_data = torch.load(
        "get_graph_data/data/pyg_graph_datas_117_test.pt",
        weights_only=False
    )
    print(f"测试图数量: {len(test_data)}")
    
    # 统计正负样本
    total_pos = 0
    total_neg = 0
    for data in test_data:
        total_pos += (data.y == 1).sum().item()
        total_neg += (data.y == 0).sum().item()
    print(f"正类节点: {total_pos}, 负类节点: {total_neg}")
    
    # ===================================================== 评估 =====================================================
    print("\n===== 评估结果 =====")
    test_preds, test_labels = eval_model(model, test_data, device)
    
    test_auc = roc_auc_score(test_labels, test_preds)
    test_ap = average_precision_score(test_labels, test_preds)
    test_mcc = matthews_corrcoef(test_labels, (test_preds > 0.5).astype(int))
    
    print(f"Test ROC_AUC: {test_auc:.4f}")
    print(f"Test AP:      {test_ap:.4f}")
    print(f"Test MCC:     {test_mcc:.4f}")
