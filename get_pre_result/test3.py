"""
test3.py: 对比评估模块
对比增强前后的效果，验证 VGMM + 置信度筛选策略是否有效
"""

import sys
sys.path.insert(0, '.')

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from copy import deepcopy

# =====================================================
# 导入 test2.py 的函数
# =====================================================
from test2 import (
    train_vgmm,
    train_node_gan_wgangp,
    train_edge_builder,
    augment_graph_with_confidence,
    VGMMValidator
)

# =====================================================
# 下游分类模型（与 predict.py 相同）
# =====================================================
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, h_struct, h_seq):
        g = torch.sigmoid(self.gate(torch.cat([h_struct, h_seq], dim=-1)))
        return g * h_struct + (1.0 - g) * h_seq

class RBPGraphNet(nn.Module):
    def __init__(self, struct_dim, seq_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)
        self.seq_proj    = nn.Linear(seq_dim, hidden_dim)
        self.fusion = GatedFusion(hidden_dim)
        
        from torch_geometric.nn import SAGEConv
        self.convs = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x_s = F.relu(self.struct_proj(data.struct_feat))
        x_q = F.relu(self.seq_proj(data.seq_feat))
        x = self.fusion(x_s, x_q)
        
        for conv in self.convs:
            x = F.relu(conv(x, data.edge_index))
        
        return self.classifier(x).squeeze(-1)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for data in loader:
        data = data.to(device)
        logits = model(data)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu())
        all_labels.append(data.y.cpu())
    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return all_probs, all_labels


def train_downstream_model(train_data, val_data, device, epochs=50, lr=1e-3):
    """训练下游分类器，返回验证集最佳指标"""
    
    struct_dim = train_data[0].struct_feat.size(1)
    seq_dim = train_data[0].seq_feat.size(1)
    
    model = RBPGraphNet(
        struct_dim=struct_dim,
        seq_dim=seq_dim,
        hidden_dim=128,
        num_layers=3
    ).to(device)
    
    # 统计训练集正负样本
    all_labels = torch.cat([d.y for d in train_data])
    pos_num = (all_labels == 1).sum().item()
    neg_num = (all_labels == 0).sum().item()
    pos_weight = neg_num / pos_num if pos_num > 0 else 1.0
    
    print(f"    正样本: {pos_num}, 负样本: {neg_num}, pos_weight: {pos_weight:.2f}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
    
    best_val_mcc = -1.0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_probs, val_labels = eval_epoch(model, val_loader, device)
        
        val_auc = roc_auc_score(val_labels, val_probs)
        val_preds = (val_probs >= 0.5).astype(int)
        val_mcc = matthews_corrcoef(val_labels, val_preds)
        
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_val_auc = val_auc
            best_state = deepcopy(model.state_dict())
        
        if epoch % 10 == 0:
            print(f"    Epoch {epoch:02d} | Loss {train_loss:.4f} | Val AUC {val_auc:.4f} | Val MCC {val_mcc:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(best_state)
    return model, best_val_auc, best_val_mcc


def main():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # =====================================================
    # 加载数据
    # =====================================================
    print("\n[1] Loading data...")
    data_list = torch.load(
        "get_graph_data/data/pyg_graph_datas_495_train.pt",
        weights_only=False
    )
    print(f"    Total graphs: {len(data_list)}")
    
    # =====================================================
    # 划分数据集（保持划分一致）
    # =====================================================
    n_total = len(data_list)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    all_train_data = data_list[:n_train]
    val_data = data_list[n_train:n_train + n_val]
    test_data = data_list[n_train + n_val:]
    
    print(f"    Train: {len(all_train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # =====================================================
    # 方案A: Baseline（不增强）
    # =====================================================
    print("\n" + "="*60)
    print("[方案A] Baseline: 不使用数据增强")
    print("="*60)
    
    baseline_model, baseline_auc, baseline_mcc = train_downstream_model(
        train_data=all_train_data,
        val_data=val_data,
        device=device,
        epochs=50
    )
    
    print(f"\n>>> Baseline 结果: AUC={baseline_auc:.4f}, MCC={baseline_mcc:.4f}")
    
    # =====================================================
    # 方案B: 增强（使用 WGAN + VGMM + Edge Builder）
    # =====================================================
    print("\n" + "="*60)
    print("[方案B] 增强: WGAN + VGMM筛选 + 置信度边预测")
    print("="*60)
    
    # Stage 1: 训练 VGMM
    print("\n[Stage 1] Training VGMM...")
    vgmm_validator = train_vgmm(
        data_list=all_train_data,
        device=device,
        n_components=2,
        threshold=0.3,
        n_pca=64
    )
    
    # Stage 2: 训练 Edge Builder
    print("\n[Stage 2] Training Edge Builder...")
    edge_builder = train_edge_builder(
        data_list=all_train_data,
        device=device,
        epochs=30,
        lr=1e-3
    )
    edge_builder.eval()
    for p in edge_builder.parameters():
        p.requires_grad = False
    
    # Stage 3: 训练 WGAN
    print("\n[Stage 3] Training WGAN...")
    G = train_node_gan_wgangp(
        data_list=all_train_data,
        device=device,
        epochs=100,
        batch_size=64,
        lambda_gp=10.0,
        lambda_anchor=0.0
    )
    G.eval()
    for p in G.parameters():
        p.requires_grad = False
    
    # Stage 4: 对训练集进行增强
    print("\n[Stage 4] Augmenting training graphs...")
    enhanced_train = []
    for idx, data in enumerate(all_train_data):
        data_aug = augment_graph_with_confidence(
            data=deepcopy(data),
            G=G,
            edge_model=edge_builder,
            device=device,
            target_ratio=1.0,
            vgmm_validator=vgmm_validator,
            max_retry=3,
            vgmm_threshold=0.3
        )
        enhanced_train.append(data_aug)
        if (idx + 1) % 50 == 0:
            print(f"    Augmented {idx+1}/{len(all_train_data)} graphs")
    
    print(f"    增强完成，生成 {len(enhanced_train)} 个图")
    
    # Stage 5: 训练下游模型
    print("\n[Stage 5] Training downstream model on enhanced data...")
    enhanced_model, enhanced_auc, enhanced_mcc = train_downstream_model(
        train_data=enhanced_train,
        val_data=val_data,
        device=device,
        epochs=50
    )
    
    print(f"\n>>> Enhanced 结果: AUC={enhanced_auc:.4f}, MCC={enhanced_mcc:.4f}")
    
    # =====================================================
    # 对比总结
    # =====================================================
    print("\n" + "="*60)
    print("对比总结")
    print("="*60)
    print(f"Baseline:  AUC={baseline_auc:.4f}, MCC={baseline_mcc:.4f}")
    print(f"Enhanced:  AUC={enhanced_auc:.4f}, MCC={enhanced_mcc:.4f}")
    print(f"提升:       AUC delta={enhanced_auc-baseline_auc:+.4f}, MCC delta={enhanced_mcc-baseline_mcc:+.4f}")
    
    if enhanced_mcc > baseline_mcc:
        print("\n结论: 增强策略有效！")
    else:
        print("\n结论: 增强策略未带来提升，需要调优")


if __name__ == "__main__":
    main()
