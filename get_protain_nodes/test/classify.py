import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score
)
import os


TRAIN_PATH = r"get_protain_nodes\\test\\data\\enhanced_train_graphs.pt"
VAL_PATH = r"get_protain_nodes\\test\\data\\val_graphs.pt"
TEST_PATH = r"get_graph_data\\data\\pyg_graph_datas_117_test.pt"
OUTPUT_DIR = r"get_protain_nodes\\test"

SEED = 42
EPOCHS = 50
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 128
NUM_LAYERS = 3
HEADS = 4
DROPOUT = 0.3


class RBPGraphNetGAT(nn.Module):
    def __init__(
        self, 
        node_feature_dim, 
        hidden_dim=128, 
        num_layers=3,
        heads=4,
        dropout=0.3
    ):
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i < num_layers - 1:
                self.convs.append(
                    GATConv(
                        hidden_dim, 
                        hidden_dim // heads, 
                        heads=heads,
                        dropout=dropout,
                        concat=True
                    )
                )
            else:
                self.convs.append(
                    GATConv(
                        hidden_dim, 
                        hidden_dim, 
                        heads=1,
                        dropout=dropout,
                        concat=False
                    )
                )
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x = data.x
        
        x = self.input_proj(x)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, data.edge_index)
            
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = F.elu(x)
                x = self.dropout(x)
        
        return self.classifier(x).squeeze(-1)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y.float())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu())
        all_labels.append(batch.y.cpu())

    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return all_probs, all_labels


def compute_metrics(labels, probs):
    preds = (probs >= 0.5).astype(int)
    
    roc_auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    mcc = matthews_corrcoef(labels, preds)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'mcc': mcc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    print("=" * 50)
    print("加载数据")
    print("=" * 50)
    
    # 加载增强后的训练集
    print(f"加载增强训练图: {TRAIN_PATH}")
    train_data_orig = torch.load(TRAIN_PATH, weights_only=False, map_location='cpu')
    train_data = list(train_data_orig)
    del train_data_orig
    print(f"训练图数量: {len(train_data)}")
    
    # 加载原始验证集（不增强）
    print(f"加载原始验证图: {VAL_PATH}")
    val_data_orig = torch.load(VAL_PATH, weights_only=False, map_location='cpu')
    val_data = list(val_data_orig)
    del val_data_orig
    print(f"验证图数量: {len(val_data)}")
    
    # 加载测试集
    print(f"加载测试图: {TEST_PATH}")
    test_data_orig = torch.load(TEST_PATH, weights_only=False, map_location='cpu')
    test_data = list(test_data_orig)
    del test_data_orig
    print(f"测试图数量: {len(test_data)}")
    
    node_feature_dim = train_data[0].x.size(1)
    print(f"节点特征维度: {node_feature_dim}")
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_data)} (已增强)")
    print(f"  验证集: {len(val_data)} (原始未增强)")
    print(f"  测试集: {len(test_data)}")

    # 重建Data对象（解决PyG 2.x的collate问题）
    def rebuild_data(data):
        new_data = Data(
            x=data.x,
            edge_index=data.edge_index,
            y=data.y,
            num_nodes=data.num_nodes
        )
        if hasattr(data, 'pos') and data.pos is not None:
            new_data.pos = data.pos
        return new_data

    train_data = [rebuild_data(d) for d in train_data]
    val_data = [rebuild_data(d) for d in val_data]
    test_data = [rebuild_data(d) for d in test_data]

    # 确保所有数据在CPU上并规范化属性
    def normalize_data(data):
        if not isinstance(data.x, torch.Tensor):
            data.x = torch.tensor(data.x, dtype=torch.float)
        if data.x.device.type != 'cpu':
            data.x = data.x.cpu()
            
        if not isinstance(data.y, torch.Tensor):
            data.y = torch.tensor(data.y, dtype=torch.long)
        if data.y.device.type != 'cpu':
            data.y = data.y.cpu()
            
        if data.edge_index is not None:
            if not isinstance(data.edge_index, torch.Tensor):
                data.edge_index = torch.tensor(data.edge_index, dtype=torch.long)
            if data.edge_index.device.type != 'cpu':
                data.edge_index = data.edge_index.cpu()
        
        # 确保num_nodes存在
        if not hasattr(data, 'num_nodes') or data.num_nodes is None:
            data.num_nodes = data.x.size(0)
        
        # 处理pos属性
        if hasattr(data, 'pos') and data.pos is not None:
            if data.pos.device.type != 'cpu':
                data.pos = data.pos.cpu()
        
        return data
    
    train_data = [normalize_data(d) for d in train_data]
    val_data = [normalize_data(d) for d in val_data]
    test_data = [normalize_data(d) for d in test_data]
    
    # 检查训练集是否有问题
    missing_num_nodes = []
    for i, d in enumerate(train_data):
        if not hasattr(d, 'num_nodes') or d.num_nodes is None:
            missing_num_nodes.append(i)
            print(f"训练集 {i}: {list(d.__dict__.keys())[:10]}")
    if missing_num_nodes:
        print(f"警告: 训练集中以下索引缺少num_nodes: {missing_num_nodes[:10]}")
    
    # 检查验证集
    for i, d in enumerate(val_data):
        if not hasattr(d, 'num_nodes') or d.num_nodes is None:
            print(f"警告: 验证集 {i} 缺少num_nodes")
    
    # 检查测试集
    for i, d in enumerate(test_data):
        if not hasattr(d, 'num_nodes') or d.num_nodes is None:
            print(f"警告: 测试集 {i} 缺少num_nodes")

    # 使用默认collate
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    def count_pos_neg(data_list):
        all_labels = torch.cat([d.y for d in data_list])
        pos_num = (all_labels == 1).sum().item()
        neg_num = (all_labels == 0).sum().item()
        return pos_num, neg_num

    pos_num, neg_num = count_pos_neg(train_data)
    pos_weight = neg_num / pos_num if pos_num > 0 else 1.0
    print(f"\n训练集统计:")
    print(f"  正样本: {pos_num}, 负样本: {neg_num}")
    print(f"  正样本权重: {pos_weight:.2f}")

    model = RBPGraphNetGAT(
        node_feature_dim=node_feature_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        heads=HEADS,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight).to(device)
    )

    best_val_mcc = -1.0
    best_val_auc = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, "best_gnn_model.pt")
    patience = 10
    counter = 0

    print("\n" + "=" * 50)
    print("开始训练")
    print("=" * 50)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_probs, val_labels = eval_epoch(model, val_loader, device)
        val_metrics = compute_metrics(val_labels, val_probs)

        if val_metrics['mcc'] > best_val_mcc:
            best_val_mcc = val_metrics['mcc']
            best_val_auc = val_metrics['roc_auc']
            counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            counter += 1

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {train_loss:.4f} | "
            f"Val AUC {val_metrics['roc_auc']:.4f} | "
            f"Val MCC {val_metrics['mcc']:.4f} | "
            f"Val F1 {val_metrics['f1']:.4f}"
        )
        
        if counter >= patience:
            print(f"早停: 验证集MCC连续{patience}轮未提升")
            break

    print("\n" + "=" * 50)
    print("测试评估")
    print("=" * 50)
    
    model.load_state_dict(torch.load(best_model_path))
    test_probs, test_labels = eval_epoch(model, test_loader, device)
    test_metrics = compute_metrics(test_labels, test_probs)

    print(f"\n测试集结果:")
    print(f"  ROC_AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  PR_AUC:    {test_metrics['pr_auc']:.4f}")
    print(f"  MCC:       {test_metrics['mcc']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    
    print(f"\n最佳模型已保存到: {best_model_path}")


if __name__ == "__main__":
    main()