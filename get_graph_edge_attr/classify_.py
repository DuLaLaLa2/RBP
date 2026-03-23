import torch
import numpy as np
from pathlib import Path
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score
)
from model import DeepResidualGAT, FocalLoss

# ====================== 超参数 ======================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IN_DIM = 1408          # 节点特征维度
HIDDEN_DIM = 128       # 隐藏维度
OUT_DIM = 2            # 分类数
NUM_LAYERS = 4
HEADS = 4
DROPOUT = 0.6
EDGE_DIM = 5           # 边特征维度
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 100
GAMMA = 2.0            # FocalLoss核心参数

# ====================== 数据加载 ======================
def load_data():
    data_path = Path("get_graph_edge_attr/data/pyg_graph_datas_495_train_edge_attr.pt")
    data_path2 = Path("get_graph_edge_attr/data/pyg_graph_datas_117_train_edge_attr.pt")
    data_train_list = torch.load(data_path,weights_only=False)
    data_test_list = torch.load(data_path2,weights_only=False)


    np.random.seed(42)
    idx = np.random.permutation(len(data_train_list))
    train_idx = idx[:int(0.8 * len(idx))]
    val_idx = idx[int(0.8 * len(idx)):]
    

    train_data = [data_train_list[i] for i in train_idx]
    val_data = [data_train_list[i] for i in val_idx]
    test_data = data_test_list

    return train_data, val_data, test_data

# ====================== 训练函数 ======================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

# ====================== 评估函数（返回6大指标）======================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds_all = []
    probs_all = []
    labels_all = []

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        
        # 预测概率（用于AUC计算）
        prob = torch.softmax(out, dim=1)[:, 1]  # 取正类概率
        # 预测标签
        pred = out.argmax(dim=1)

        probs_all.extend(prob.cpu().numpy())
        preds_all.extend(pred.cpu().numpy())
        labels_all.extend(data.y.cpu().numpy())

    # 计算6个核心指标
    roc_auc = roc_auc_score(labels_all, probs_all)
    pr_auc = average_precision_score(labels_all, probs_all)
    mcc = matthews_corrcoef(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average='binary')
    precision = precision_score(labels_all, preds_all, zero_division=0)
    recall = recall_score(labels_all, preds_all, zero_division=0)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mcc": mcc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# ====================== 主训练流程 ======================
if __name__ == "__main__":
    # 数据
    train_data, val_data, test_data = load_data()
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # 模型
    model = DeepResidualGAT(
        in_dim=IN_DIM,
        hidden_dim=HIDDEN_DIM,
        out_dim=OUT_DIM,
        num_layers=NUM_LAYERS,
        heads=HEADS,
        dropout=DROPOUT,
        edge_dim=EDGE_DIM
    ).to(DEVICE)

    # 损失函数 & 优化器
    criterion = FocalLoss(gamma=GAMMA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    # 训练
    best_mcc = 0.0  # 蛋白质任务推荐用MCC作为最优模型指标
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)

        # 根据MCC保存最优模型
        if val_metrics['mcc'] > best_mcc:
            best_mcc = val_metrics['mcc']
            torch.save(model.state_dict(), "get_graph_edge_attr/best_model.pt")

        print(f"Epoch {epoch:03d} | Loss {train_loss:.4f}")
        print(f"  Val ROC_AUC: {val_metrics['roc_auc']:.4f}")
        print(f"  Val PR_AUC:  {val_metrics['pr_auc']:.4f}")
        print(f"  Val MCC:     {val_metrics['mcc']:.4f}")
        print(f"  Val F1:      {val_metrics['f1']:.4f}")

    # 测试集评估（输出你指定的格式）
    model.load_state_dict(torch.load("get_graph_edge_attr/best_model.pt", weights_only=False))
    test_metrics = evaluate(model, test_loader, DEVICE)
    
    print("\n==================== Test Results ====================")
    print(f"  ROC_AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  PR_AUC:    {test_metrics['pr_auc']:.4f}")
    print(f"  MCC:       {test_metrics['mcc']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
