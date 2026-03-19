import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from copy import deepcopy
import random
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

# 基线比较：不做数据增强只在GAT上训练与分类
# ===================================================== GAT 分类器 (与GAN-GNN.py一致) =====================================================
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


# ===================================================== Focal Loss =====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


# ===================================================== 训练函数 =====================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_preds.append(probs.cpu())
        all_labels.append(data.y.cpu())
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


# ===================================================== 主函数 (不使用WGAN增强) =====================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    
    # ===================================================== 加载495训练数据 =====================================================
    print("\n===== 加载495训练数据 =====")
    train_val_data = torch.load(
        "get_graph_data/data/pyg_graph_datas_495_train.pt",
        weights_only=False
    )
    
    # 划分train/val
    random.shuffle(train_val_data)
    n_total = len(train_val_data)
    n_train = int(0.7 * n_total)
    train_data = train_val_data[:n_train]
    val_data = train_val_data[n_train:]
    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}")
    
    # ===================================================== 加载117测试数据 =====================================================
    print("\n===== 加载117测试数据 =====")
    test_data = torch.load(
        "get_graph_data/data/pyg_graph_datas_117_test.pt",
        weights_only=False
    )
    print(f"测试图数量: {len(test_data)}")
    
    # 统计正负样本
    total_pos = sum((d.y == 1).sum().item() for d in test_data)
    total_neg = sum((d.y == 0).sum().item() for d in test_data)
    print(f"测试集: 正类={total_pos}, 负类={total_neg}")
    
    # ===================================================== 创建DataLoader =====================================================
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
    
    # ===================================================== 模型 =====================================================
    model = GATClassifier(in_dim=1408, hidden_dim=256, out_dim=2, heads=4).to(device)
    
    # Focal Loss
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # ===================================================== 训练 =====================================================
    print("\n===== 训练中 =====")
    best_val_mcc = -1.0
    best_model_state = None
    
    for epoch in range(1, 51):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # 验证集
        val_preds, val_labels = eval_model(model, val_loader, device)
        val_auc = roc_auc_score(val_labels, val_preds)
        val_mcc = matthews_corrcoef(val_labels, (val_preds > 0.5).astype(int))
        
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_model_state = deepcopy(model.state_dict())
        
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | Val MCC: {val_mcc:.4f}")
    
    # ===================================================== 加载最佳模型测试 =====================================================
    print("\n===== 测试结果 =====")
    model.load_state_dict(best_model_state)
    
    # 验证集最终结果
    val_preds, val_labels = eval_model(model, val_loader, device)
    val_auc = roc_auc_score(val_labels, val_preds)
    val_ap = average_precision_score(val_labels, val_preds)
    val_mcc = matthews_corrcoef(val_labels, (val_preds > 0.5).astype(int))
    
    print(f"验证集 - AUC: {val_auc:.4f} | AP: {val_ap:.4f} | MCC: {val_mcc:.4f}")
    
    # 测试集结果
    test_preds, test_labels = eval_model(model, test_loader, device)
    test_auc = roc_auc_score(test_labels, test_preds)
    test_ap = average_precision_score(test_labels, test_preds)
    test_mcc = matthews_corrcoef(test_labels, (test_preds > 0.5).astype(int))
    
    print(f"测试集 - AUC: {test_auc:.4f} | AP: {test_ap:.4f} | MCC: {test_mcc:.4f}")
    
    # 保存模型
    torch.save(best_model_state, "get_pre_result/data/gat_baseline.pt")
    print("\n模型已保存: gat_baseline.pt")
'''
===== 训练中 =====
Epoch 01 | Loss: 0.1036 | Val AUC: 0.7979 | Val MCC: 0.3625
Epoch 02 | Loss: 0.0652 | Val AUC: 0.8341 | Val MCC: 0.3733
Epoch 03 | Loss: 0.0631 | Val AUC: 0.8405 | Val MCC: 0.3624
Epoch 04 | Loss: 0.0615 | Val AUC: 0.8460 | Val MCC: 0.4040
Epoch 05 | Loss: 0.0587 | Val AUC: 0.8489 | Val MCC: 0.4238
Epoch 06 | Loss: 0.0570 | Val AUC: 0.8506 | Val MCC: 0.3844
Epoch 07 | Loss: 0.0567 | Val AUC: 0.8440 | Val MCC: 0.3677
Epoch 08 | Loss: 0.0566 | Val AUC: 0.8521 | Val MCC: 0.4313
Epoch 09 | Loss: 0.0545 | Val AUC: 0.8452 | Val MCC: 0.4236
Epoch 10 | Loss: 0.0535 | Val AUC: 0.8459 | Val MCC: 0.3500
Epoch 11 | Loss: 0.0534 | Val AUC: 0.8446 | Val MCC: 0.4050
Epoch 12 | Loss: 0.0518 | Val AUC: 0.8457 | Val MCC: 0.4132
Epoch 13 | Loss: 0.0505 | Val AUC: 0.8470 | Val MCC: 0.4030
Epoch 14 | Loss: 0.0500 | Val AUC: 0.8421 | Val MCC: 0.3957
Epoch 15 | Loss: 0.0502 | Val AUC: 0.8448 | Val MCC: 0.3947
Epoch 16 | Loss: 0.0498 | Val AUC: 0.8476 | Val MCC: 0.3876
Epoch 17 | Loss: 0.0483 | Val AUC: 0.8464 | Val MCC: 0.3836
Epoch 18 | Loss: 0.0470 | Val AUC: 0.8464 | Val MCC: 0.3584
Epoch 19 | Loss: 0.0475 | Val AUC: 0.8428 | Val MCC: 0.4180
Epoch 20 | Loss: 0.0472 | Val AUC: 0.8461 | Val MCC: 0.4201
Epoch 21 | Loss: 0.0468 | Val AUC: 0.8472 | Val MCC: 0.3979
Epoch 22 | Loss: 0.0483 | Val AUC: 0.8406 | Val MCC: 0.3960
Epoch 23 | Loss: 0.0461 | Val AUC: 0.8519 | Val MCC: 0.4074
Epoch 24 | Loss: 0.0437 | Val AUC: 0.8478 | Val MCC: 0.3878
Epoch 25 | Loss: 0.0444 | Val AUC: 0.8434 | Val MCC: 0.4126
Epoch 26 | Loss: 0.0436 | Val AUC: 0.8439 | Val MCC: 0.4282
Epoch 27 | Loss: 0.0448 | Val AUC: 0.8444 | Val MCC: 0.3901
Epoch 28 | Loss: 0.0433 | Val AUC: 0.8391 | Val MCC: 0.3441
Epoch 29 | Loss: 0.0437 | Val AUC: 0.8445 | Val MCC: 0.3972
Epoch 30 | Loss: 0.0448 | Val AUC: 0.8411 | Val MCC: 0.3763
Epoch 31 | Loss: 0.0439 | Val AUC: 0.8357 | Val MCC: 0.3826
Epoch 32 | Loss: 0.0411 | Val AUC: 0.8307 | Val MCC: 0.3991
Epoch 33 | Loss: 0.0405 | Val AUC: 0.8435 | Val MCC: 0.3630
Epoch 34 | Loss: 0.0411 | Val AUC: 0.8432 | Val MCC: 0.4023
Epoch 35 | Loss: 0.0479 | Val AUC: 0.8384 | Val MCC: 0.3875
Epoch 36 | Loss: 0.0443 | Val AUC: 0.8435 | Val MCC: 0.3777
Epoch 37 | Loss: 0.0408 | Val AUC: 0.8436 | Val MCC: 0.3718
Epoch 38 | Loss: 0.0401 | Val AUC: 0.8450 | Val MCC: 0.3986
Epoch 39 | Loss: 0.0407 | Val AUC: 0.8422 | Val MCC: 0.3675
Epoch 40 | Loss: 0.0398 | Val AUC: 0.8407 | Val MCC: 0.3984
Epoch 41 | Loss: 0.0413 | Val AUC: 0.8410 | Val MCC: 0.3763
Epoch 42 | Loss: 0.0406 | Val AUC: 0.8436 | Val MCC: 0.3679
Epoch 43 | Loss: 0.0412 | Val AUC: 0.8391 | Val MCC: 0.3744
Epoch 44 | Loss: 0.0396 | Val AUC: 0.8425 | Val MCC: 0.3415
Epoch 45 | Loss: 0.0410 | Val AUC: 0.8392 | Val MCC: 0.3780
Epoch 46 | Loss: 0.0410 | Val AUC: 0.8302 | Val MCC: 0.2720
Epoch 47 | Loss: 0.0406 | Val AUC: 0.8441 | Val MCC: 0.4126
Epoch 48 | Loss: 0.0396 | Val AUC: 0.8427 | Val MCC: 0.4103
Epoch 49 | Loss: 0.0395 | Val AUC: 0.8462 | Val MCC: 0.4011
Epoch 50 | Loss: 0.0389 | Val AUC: 0.8381 | Val MCC: 0.4040

===== 测试结果 =====
验证集 - AUC: 0.8521 | AP: 0.5092 | MCC: 0.4313
测试集 - AUC: 0.8157 | AP: 0.2265 | MCC: 0.2543
'''