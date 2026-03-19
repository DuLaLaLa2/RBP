import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader
from copy import deepcopy
import random
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef


# ===================================================== GraphSAGE 分类器 =====================================================
class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_dim: int = 1408, hidden_dim: int = 256, out_dim: int = 2, num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
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


# ===================================================== 主函数 =====================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    
    # ===================================================== 加载495数据 =====================================================
    print("\n===== 加载495数据 =====")
    train_val_data = torch.load(
        "get_graph_data/data/pyg_graph_datas_495_train.pt",
        weights_only=False
    )
    
    def is_valid(graph):
        pos_idx = torch.where(graph.y == 1)[0]
        if graph.x.size(0) > 500:
            return False
        return len(pos_idx) >= 2 and graph.edge_index.size(1) >= 2 and graph.x.size(0) >= 2
    
    train_val_data = [g for g in train_val_data if is_valid(g)]
    print(f"有效图数量: {len(train_val_data)}")
    
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
    
    total_pos = sum((d.y == 1).sum().item() for d in test_data)
    total_neg = sum((d.y == 0).sum().item() for d in test_data)
    print(f"测试集: 正类={total_pos}, 负类={total_neg}")
    
    # ===================================================== DataLoader =====================================================
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
    
    # ===================================================== 模型 =====================================================
    model = GraphSAGEClassifier(in_dim=1408, hidden_dim=256, out_dim=2, num_layers=3, dropout=0.3).to(device)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # ===================================================== 训练 =====================================================
    print("\n===== 训练GraphSAGE =====")
    best_val_mcc = -1.0
    best_model_state = None
    
    for epoch in range(1, 51):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        val_preds, val_labels = eval_model(model, val_loader, device)
        val_auc = roc_auc_score(val_labels, val_preds)
        val_mcc = matthews_corrcoef(val_labels, (val_preds > 0.5).astype(int))
        
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_model_state = deepcopy(model.state_dict())
        
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | Val MCC: {val_mcc:.4f}")
    
    # ===================================================== 测试 =====================================================
    print("\n===== 测试结果 =====")
    model.load_state_dict(best_model_state)
    
    val_preds, val_labels = eval_model(model, val_loader, device)
    val_auc = roc_auc_score(val_labels, val_preds)
    val_mcc = matthews_corrcoef(val_labels, (val_preds > 0.5).astype(int))
    
    print(f"验证集 - AUC: {val_auc:.4f} | MCC: {val_mcc:.4f}")
    
    test_preds, test_labels = eval_model(model, test_loader, device)
    test_auc = roc_auc_score(test_labels, test_preds)
    test_mcc = matthews_corrcoef(test_labels, (test_preds > 0.5).astype(int))
    
    print(f"测试集 - AUC: {test_auc:.4f} | MCC: {test_mcc:.4f}")
    
    # 保存模型
    torch.save(best_model_state, "get_pre_result/data/graphsage_baseline.pt")
    print("\n模型已保存: graphsage_baseline.pt")

    '''
    ===== 加载495数据 =====
有效图数量: 430
训练集: 301, 验证集: 129

===== 加载117测试数据 =====
测试图数量: 117
测试集: 正类=2031, 负类=35314

===== 训练GraphSAGE =====
Epoch 01 | Loss: 0.0769 | Val AUC: 0.8374 | Val MCC: 0.4064
Epoch 02 | Loss: 0.0618 | Val AUC: 0.8539 | Val MCC: 0.4383
Epoch 03 | Loss: 0.0595 | Val AUC: 0.8607 | Val MCC: 0.3181
Epoch 04 | Loss: 0.0554 | Val AUC: 0.8650 | Val MCC: 0.4313
Epoch 05 | Loss: 0.0528 | Val AUC: 0.8615 | Val MCC: 0.3655
Epoch 06 | Loss: 0.0514 | Val AUC: 0.8660 | Val MCC: 0.4625
Epoch 07 | Loss: 0.0486 | Val AUC: 0.8596 | Val MCC: 0.4124
Epoch 08 | Loss: 0.0474 | Val AUC: 0.8589 | Val MCC: 0.4646
Epoch 09 | Loss: 0.0435 | Val AUC: 0.8576 | Val MCC: 0.4548
Epoch 10 | Loss: 0.0441 | Val AUC: 0.8555 | Val MCC: 0.4582
Epoch 11 | Loss: 0.0406 | Val AUC: 0.8535 | Val MCC: 0.4482
Epoch 12 | Loss: 0.0391 | Val AUC: 0.8515 | Val MCC: 0.4435
Epoch 13 | Loss: 0.0397 | Val AUC: 0.8478 | Val MCC: 0.3925
Epoch 14 | Loss: 0.0386 | Val AUC: 0.8540 | Val MCC: 0.4487
Epoch 15 | Loss: 0.0377 | Val AUC: 0.8596 | Val MCC: 0.4579
Epoch 16 | Loss: 0.0340 | Val AUC: 0.8518 | Val MCC: 0.4485
Epoch 17 | Loss: 0.0327 | Val AUC: 0.8463 | Val MCC: 0.4407
Epoch 18 | Loss: 0.0310 | Val AUC: 0.8496 | Val MCC: 0.4494
Epoch 19 | Loss: 0.0297 | Val AUC: 0.8482 | Val MCC: 0.4313
Epoch 20 | Loss: 0.0295 | Val AUC: 0.8490 | Val MCC: 0.4479
Epoch 21 | Loss: 0.0292 | Val AUC: 0.8481 | Val MCC: 0.4278
Epoch 22 | Loss: 0.0270 | Val AUC: 0.8475 | Val MCC: 0.4353
Epoch 23 | Loss: 0.0262 | Val AUC: 0.8502 | Val MCC: 0.4503
Epoch 24 | Loss: 0.0263 | Val AUC: 0.8457 | Val MCC: 0.4330
Epoch 25 | Loss: 0.0237 | Val AUC: 0.8484 | Val MCC: 0.4345
Epoch 26 | Loss: 0.0231 | Val AUC: 0.8485 | Val MCC: 0.4425
Epoch 27 | Loss: 0.0221 | Val AUC: 0.8442 | Val MCC: 0.4209
Epoch 28 | Loss: 0.0224 | Val AUC: 0.8498 | Val MCC: 0.4486
Epoch 29 | Loss: 0.0204 | Val AUC: 0.8435 | Val MCC: 0.4215
Epoch 30 | Loss: 0.0201 | Val AUC: 0.8472 | Val MCC: 0.4277
Epoch 31 | Loss: 0.0196 | Val AUC: 0.8437 | Val MCC: 0.4317
Epoch 32 | Loss: 0.0195 | Val AUC: 0.8489 | Val MCC: 0.4427
Epoch 33 | Loss: 0.0185 | Val AUC: 0.8480 | Val MCC: 0.4418
Epoch 34 | Loss: 0.0179 | Val AUC: 0.8466 | Val MCC: 0.4325
Epoch 35 | Loss: 0.0167 | Val AUC: 0.8431 | Val MCC: 0.4169
Epoch 36 | Loss: 0.0172 | Val AUC: 0.8479 | Val MCC: 0.4432
Epoch 37 | Loss: 0.0166 | Val AUC: 0.8428 | Val MCC: 0.4358
Epoch 38 | Loss: 0.0148 | Val AUC: 0.8480 | Val MCC: 0.4351
Epoch 39 | Loss: 0.0160 | Val AUC: 0.8473 | Val MCC: 0.4379
Epoch 40 | Loss: 0.0160 | Val AUC: 0.8410 | Val MCC: 0.4248
Epoch 41 | Loss: 0.0164 | Val AUC: 0.8444 | Val MCC: 0.4275
Epoch 42 | Loss: 0.0148 | Val AUC: 0.8424 | Val MCC: 0.4146
Epoch 43 | Loss: 0.0138 | Val AUC: 0.8466 | Val MCC: 0.4278
Epoch 44 | Loss: 0.0142 | Val AUC: 0.8387 | Val MCC: 0.4156
Epoch 45 | Loss: 0.0145 | Val AUC: 0.8523 | Val MCC: 0.4467
Epoch 46 | Loss: 0.0137 | Val AUC: 0.8466 | Val MCC: 0.4242
Epoch 47 | Loss: 0.0121 | Val AUC: 0.8515 | Val MCC: 0.4363
Epoch 48 | Loss: 0.0120 | Val AUC: 0.8473 | Val MCC: 0.4301
Epoch 49 | Loss: 0.0119 | Val AUC: 0.8485 | Val MCC: 0.4291
Epoch 50 | Loss: 0.0118 | Val AUC: 0.8461 | Val MCC: 0.4268

===== 测试结果 =====
验证集 - AUC: 0.8589 | MCC: 0.4646
测试集 - AUC: 0.8271 | MCC: 0.2477
    '''
