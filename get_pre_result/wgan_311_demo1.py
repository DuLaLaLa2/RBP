import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import negative_sampling
from copy import deepcopy
from typing import List, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
import numpy as np


# ===================================================== 1. WGAN-GP (使用统计差异作为条件) =====================================================
class WGANGenerator(nn.Module):
    """
    WGAN生成器 - 条件 = 标签 + 正负类统计差异
    """
    def __init__(self, noise_dim: int = 64, cond_dim: int = 32 + 128, feat_dim: int = 1408, hidden_dim: int = 512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, feat_dim)
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.backbone(torch.cat([z, cond], dim=-1))


class WGANDiscriminator(nn.Module):
    """WGAN判别器"""
    def __init__(self, feat_dim: int = 1408, cond_dim: int = 32 + 128, hidden_dim: int = 512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(feat_dim + cond_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.backbone(torch.cat([feat, cond], dim=-1))


def gradient_penalty(discriminator, real_x: torch.Tensor, fake_x: torch.Tensor, cond: torch.Tensor, device: torch.device) -> torch.Tensor:
    """WGAN-GP梯度惩罚"""
    alpha = torch.rand(real_x.size(0), 1, device=device)
    interpolated = alpha * real_x + (1 - alpha) * fake_x
    interpolated.requires_grad_(True)
    d_interpolated = discriminator(interpolated, cond)
    grad = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones_like(d_interpolated),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad = grad.view(grad.size(0), -1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


# ===================================================== 2. 边缘预测器 =====================================================
class EdgePredictor(nn.Module):
    def __init__(self, node_dim: int = 1408, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        return self.backbone(torch.cat([h_i, h_j], dim=-1)).squeeze(-1)


# ===================================================== 3. GAT 分类器 =====================================================
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


# ===================================================== 4. Focal Loss =====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


# ===================================================== 5. 计算条件特征 (正负类统计差异) =====================================================
def compute_stat_conds(graphs: List[Data], device: torch.device) -> dict:
    """
    计算正负类的统计特征作为条件
    """
    pos_struct_list = []
    neg_struct_list = []
    
    for graph in graphs:
        graph = graph.to(device)
        pos_mask = graph.y == 1
        neg_mask = graph.y == 0
        
        if pos_mask.sum() > 0:
            pos_struct_list.append(graph.struct_feat[pos_mask])
        if neg_mask.sum() > 0:
            neg_struct_list.append(graph.struct_feat[neg_mask])
    
    pos_struct = torch.cat(pos_struct_list, dim=0) if pos_struct_list else None
    neg_struct = torch.cat(neg_struct_list, dim=0) if neg_struct_list else None
    
    # 统计差异
    if pos_struct is not None and neg_struct is not None:
        pos_mean = pos_struct.mean(dim=0)
        neg_mean = neg_struct.mean(dim=0)
        stat_diff = pos_mean - neg_mean  # 关键差异！
        stat_scale = (pos_struct.std(dim=0) + neg_struct.std(dim=0)) / 2
    elif pos_struct is not None:
        stat_diff = pos_struct.mean(dim=0)
        stat_scale = pos_struct.std(dim=0)
    else:
        stat_diff = torch.zeros(128, device=device)
        stat_scale = torch.ones(128, device=device)
    
    return {
        'pos_mean': pos_mean if pos_struct is not None else None,
        'neg_mean': neg_mean if neg_struct is not None else None,
        'stat_diff': stat_diff,
        'stat_scale': stat_scale
    }


# ===================================================== 6. 预训练阶段 =====================================================
_global_discriminator = None
_global_opt_D = None


def pretrain_gan_and_edge(
    train_graphs: List[Data],
    device: torch.device,
    gan_epochs: int = 100,
    edge_epochs: int = 50,
    noise_dim: int = 64,
    lr: float = 1e-4,
    gan_batch_size: int = 32,
    lambda_gp: float = 10.0
) -> Tuple[WGANGenerator, EdgePredictor, dict]:
    """预训练WGAN + Edge Predictor"""
    
    # 1. 计算统计条件
    stat_conds = compute_stat_conds(train_graphs, device)
    
    # 标签embedding
    label_embed = nn.Parameter(torch.randn(32)).to(device)
    
    # 条件维度 = 标签(32) + 统计差异(128)
    cond_dim = 32 + 128
    
    # 2. 收集正类特征
    pos_feats = []
    for graph in train_graphs:
        graph = graph.to(device)
        pos_mask = graph.y == 1
        pos_idx = torch.where(pos_mask)[0]
        if len(pos_idx) >= 2:
            pos_feats.append(graph.x[pos_idx])
    
    pos_feats = torch.cat(pos_feats, dim=0)
    feat_dim = pos_feats.size(1)
    
    # 3. 初始化WGAN
    generator = WGANGenerator(noise_dim, cond_dim, feat_dim).to(device)
    discriminator = WGANDiscriminator(feat_dim, cond_dim).to(device)
    opt_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
    
    # 4. 训练WGAN
    print("\n===== 预训练阶段: WGAN-GP (统计差异条件) =====")
    n_samples = pos_feats.size(0)
    steps_per_epoch = max(1, n_samples // gan_batch_size)
    
    for epoch in range(gan_epochs):
        perm = torch.randperm(n_samples)
        for step in range(steps_per_epoch):
            idx = perm[step * gan_batch_size:(step + 1) * gan_batch_size]
            real_x = pos_feats[idx].to(device)
            
            # 条件 = 标签(正类=1) + 统计差异
            batch_label = torch.ones(len(idx), 32, device=device)  # 正类标签
            batch_cond = torch.cat([batch_label, stat_conds['stat_diff'].unsqueeze(0).expand(len(idx), -1)], dim=-1)
            
            # 训练判别器
            for _ in range(5):
                z = torch.randn(real_x.size(0), noise_dim, device=device)
                fake_x = generator(z, batch_cond).detach()
                d_real = discriminator(real_x, batch_cond).mean()
                d_fake = discriminator(fake_x, batch_cond).mean()
                gp = gradient_penalty(discriminator, real_x, fake_x, batch_cond, device)
                loss_D = d_fake - d_real + lambda_gp * gp
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()
            
            # 训练生成器
            z = torch.randn(real_x.size(0), noise_dim, device=device)
            fake_x = generator(z, batch_cond)
            loss_G = -discriminator(fake_x, batch_cond).mean()
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
        
        if epoch % 20 == 0:
            print(f"[WGAN-GP] Epoch {epoch:03d} | D: {loss_D.item():.4f} | G: {loss_G.item():.4f}")
    
    # 5. 训练边缘预测器
    print("\n===== 预训练阶段: 边缘预测器 =====")
    edge_predictor = EdgePredictor(node_dim=feat_dim).to(device)
    opt_edge = torch.optim.Adam(edge_predictor.parameters(), lr=lr)
    
    for epoch in range(edge_epochs):
        total_loss = 0.0
        for graph in train_graphs:
            graph = graph.to(device)
            node_feat = graph.x
            pos_edges = graph.edge_index
            if pos_edges.size(1) < 2:
                continue
            
            neg_samples = min(pos_edges.size(1), 1000)
            neg_edges = negative_sampling(pos_edges, num_nodes=node_feat.size(0), num_neg_samples=neg_samples)
            
            def edge_loss(edges, label):
                i, j = edges
                probs = edge_predictor(node_feat[i], node_feat[j])
                return F.binary_cross_entropy(probs, torch.full_like(probs, label))
            
            loss = edge_loss(pos_edges, 1.0) + edge_loss(neg_edges, 0.0)
            opt_edge.zero_grad()
            loss.backward()
            opt_edge.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"[Edge] Epoch {epoch:03d} | Loss: {total_loss:.4f}")
    
    generator.eval()
    edge_predictor.eval()
    for p in generator.parameters():
        p.requires_grad = False
    for p in edge_predictor.parameters():
        p.requires_grad = False
    
    global _global_discriminator, _global_opt_D
    _global_discriminator = discriminator
    _global_opt_D = opt_D
    
    return generator, edge_predictor, stat_conds


# ===================================================== 7. 图增强函数 =====================================================
def augment_graph(
    graph: Data,
    generator: WGANGenerator,
    edge_predictor: EdgePredictor,
    stat_conds: dict,
    device: torch.device,
    target_ratio: float = 0.5,
    noise_dim: int = 64,
    edge_thresh: float = 0.5,
    max_new_nodes: int = 100
) -> Data:
    """图增强"""
    graph = deepcopy(graph).to(device)
    pos_mask = graph.y == 1
    n_pos = pos_mask.sum().item()
    n_neg = (graph.y == 0).sum().item()
    
    if n_pos == 0 or n_neg == 0:
        return graph
    
    target_pos = int(n_neg * target_ratio)
    n_need = max(0, min(target_pos - n_pos, max_new_nodes))
    if n_need == 0:
        return graph
    
    # 生成节点
    generator.eval()
    batch_label = torch.ones(n_need, 32, device=device)
    batch_cond = torch.cat([batch_label, stat_conds['stat_diff'].unsqueeze(0).expand(n_need, -1)], dim=-1)
    z = torch.randn(n_need, noise_dim, device=device)
    fake_feats = generator(z, batch_cond)
    
    # 扩展图
    old_n_nodes = graph.x.size(0)
    graph.x = torch.cat([graph.x, fake_feats], dim=0)
    struct_dim = graph.struct_feat.size(1)
    graph.struct_feat = torch.cat([graph.struct_feat, fake_feats[:, :struct_dim]], dim=0)
    graph.seq_feat = torch.cat([graph.seq_feat, fake_feats[:, struct_dim:]], dim=0)
    graph.y = torch.cat([graph.y, torch.ones(n_need, device=device, dtype=graph.y.dtype)], dim=0)
    
    # 边预测
    if old_n_nodes > 0:
        old_nodes = graph.x[:old_n_nodes]
        new_edges = []
        for new_idx in range(n_need):
            new_id = old_n_nodes + new_idx
            new_feat = graph.x[new_id:new_id+1]
            probs = edge_predictor(new_feat.repeat(old_n_nodes, 1), old_nodes)
            connect_mask = probs > edge_thresh
            connect_idx = torch.where(connect_mask)[0]
            for old_id in connect_idx.tolist():
                new_edges.append([new_id, old_id])
                new_edges.append([old_id, new_id])
        
        if new_edges:
            edge_add = torch.tensor(new_edges, device=device, dtype=torch.long).t()
            edge_add = edge_add[:, :min(edge_add.size(1), 5000)]
            graph.edge_index = torch.cat([graph.edge_index, edge_add], dim=1)
    
    generator.train()
    return graph


# ===================================================== 8. 两阶段训练 =====================================================
def train_full(
    train_graphs: List[Data],
    val_graphs: List[Data],
    generator: WGANGenerator,
    edge_predictor: EdgePredictor,
    stat_conds: dict,
    device: torch.device,
    gan_finetune_epochs: int = 200,
    lr: float = 1e-3,
    lambda_gan: float = 0.1,
    lambda_edge: float = 0.1
) -> GATClassifier:
    """两阶段训练"""
    
    gat = GATClassifier(in_dim=1408, hidden_dim=256, out_dim=2, heads=4).to(device)
    opt_gat = torch.optim.Adam(gat.parameters(), lr=lr)
    focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
    
    for p in generator.parameters():
        p.requires_grad = True
    for p in edge_predictor.parameters():
        p.requires_grad = True
    
    opt_G = torch.optim.Adam(generator.parameters(), lr=1e-5)
    opt_edge = torch.optim.Adam(edge_predictor.parameters(), lr=1e-5)
    opt_G_gan = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    
    discriminator = _global_discriminator
    opt_D_gan = _global_opt_D
    
    best_val_mcc = -float('inf')
    best_gat_weights = None
    
    # 阶段1: WGAN微调
    print("\n===== 阶段1: WGAN微调 (100轮) =====")
    pos_feats = []
    pos_struct = []
    for g in train_graphs:
        pos_idx = torch.where(g.y == 1)[0]
        if len(pos_idx) > 0:
            pos_feats.append(g.x[pos_idx])
    pos_feats = torch.cat(pos_feats, dim=0)
    
    for epoch in range(100):
        perm = torch.randperm(pos_feats.size(0))
        batch_size = 32
        for step in range(max(1, pos_feats.size(0) // batch_size)):
            idx = perm[step*batch_size:(step+1)*batch_size]
            real_x = pos_feats[idx].to(device)
            batch_label = torch.ones(len(idx), 32, device=device)
            batch_cond = torch.cat([batch_label, stat_conds['stat_diff'].unsqueeze(0).expand(len(idx), -1)], dim=-1)
            
            z = torch.randn(real_x.size(0), 64, device=device)
            fake_x = generator(z, batch_cond).detach()
            d_real = discriminator(real_x, batch_cond).mean()
            d_fake = discriminator(fake_x, batch_cond).mean()
            gp = gradient_penalty(discriminator, real_x, fake_x, batch_cond, device)
            loss_D = d_fake - d_real + 10.0 * gp
            
            opt_D_gan.zero_grad()
            loss_D.backward()
            opt_D_gan.step()
            
            z = torch.randn(real_x.size(0), 64, device=device)
            fake_x = generator(z, batch_cond)
            loss_G = -discriminator(fake_x, batch_cond).mean()
            opt_G_gan.zero_grad()
            loss_G.backward()
            opt_G_gan.step()
        
        if epoch % 20 == 0:
            print(f"[WGAN Finetune] Epoch {epoch:03d}")
    
    # 阶段2: GNN + 微调
    print("\n===== 阶段2: GNN + 同步微调 (200轮) =====")
    for epoch in range(gan_finetune_epochs):
        gat.train()
        generator.train()
        edge_predictor.train()
        total_loss = 0.0
        
        for graph in train_graphs:
            graph = graph.to(device)
            aug_graph = augment_graph(graph, generator, edge_predictor, stat_conds, 
                                      device, target_ratio=0.4)
            
            logits = gat(aug_graph.x, aug_graph.edge_index)
            loss_node = focal_loss(logits, aug_graph.y)
            
            pos_mask = aug_graph.y == 1
            pos_idx = torch.where(pos_mask)[0]
            if len(pos_idx) > 0:
                pos_idx = pos_idx[:min(len(pos_idx), 32)]
                real_pos = aug_graph.x[pos_idx]
                batch_label = torch.ones(len(pos_idx), 32, device=device)
                cond = torch.cat([batch_label, stat_conds['stat_diff'].unsqueeze(0).expand(len(pos_idx), -1)], dim=-1)
                z = torch.randn(len(pos_idx), 64, device=device)
                fake_pos = generator(z, cond)
                loss_gan = F.mse_loss(fake_pos, real_pos)
            else:
                loss_gan = torch.tensor(0.0, device=device)
            
            pos_edges = aug_graph.edge_index
            neg_samples = min(pos_edges.size(1), 500)
            neg_edges = negative_sampling(pos_edges, num_nodes=aug_graph.x.size(0), num_neg_samples=neg_samples)
            
            def edge_loss(edges, label):
                i, j = edges
                probs = edge_predictor(aug_graph.x[i], aug_graph.x[j])
                return F.binary_cross_entropy(probs, torch.full_like(probs, label))
            
            loss_edge = edge_loss(pos_edges, 1.0) + edge_loss(neg_edges, 0.0)
            
            loss = loss_node + lambda_gan * loss_gan + lambda_edge * loss_edge
            loss.backward()
            
            opt_gat.step()
            opt_G.step()
            opt_edge.step()
            
            opt_gat.zero_grad()
            opt_G.zero_grad()
            opt_edge.zero_grad()
            
            total_loss += loss.item()
        
        gat.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for graph in val_graphs:
                graph = graph.to(device)
                logits = gat(graph.x, graph.edge_index)
                preds = torch.softmax(logits, dim=1)[:, 1]
                val_preds.append(preds.cpu())
                val_labels.append(graph.y.cpu())
        
        val_preds = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()
        val_auc = roc_auc_score(val_labels, val_preds)
        val_mcc = matthews_corrcoef(val_labels, (val_preds > 0.5).astype(int))
        
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_gat_weights = deepcopy(gat.state_dict())
        
        print(f"[Epoch {epoch:03d}] Loss: {total_loss:.4f} | Val AUC: {val_auc:.4f} | Val MCC: {val_mcc:.4f}")
    
    if best_gat_weights is not None:
        gat.load_state_dict(best_gat_weights)
    
    return gat


# ===================================================== 主函数 =====================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    print("\n===== 加载数据 =====")
    graph_list = torch.load("get_graph_data/data/pyg_graph_datas_495_train.pt", weights_only=False)
    
    def is_valid(graph):
        pos_idx = torch.where(graph.y == 1)[0]
        if graph.x.size(0) > 500:
            return False
        return len(pos_idx) >= 2 and graph.edge_index.size(1) >= 2 and graph.x.size(0) >= 2
    
    graph_list = [g for g in graph_list if is_valid(g)]
    print(f"有效图数量: {len(graph_list)}")
    
    random.shuffle(graph_list)
    n_total = len(graph_list)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    train_graphs = graph_list[:n_train]
    val_graphs = graph_list[n_train:n_train+n_val]
    test_graphs = graph_list[n_train+n_val:]
    print(f"数据划分: Train {len(train_graphs)} | Val {len(val_graphs)} | Test {len(test_graphs)}")
    
    # 预训练
    generator, edge_predictor, stat_conds = pretrain_gan_and_edge(
        train_graphs=train_graphs,
        device=device,
        gan_epochs=100,
        edge_epochs=50,
        lr=1e-4,
        gan_batch_size=32
    )
    
    # 训练
    gat = train_full(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        generator=generator,
        edge_predictor=edge_predictor,
        stat_conds=stat_conds,
        device=device,
        gan_finetune_epochs=200,
        lr=1e-3,
        lambda_gan=0.1,
        lambda_edge=0.1
    )
    
    # 测试
    print("\n===== 测试集评估 (495内部) =====")
    gat.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for graph in test_graphs:
            graph = graph.to(device)
            logits = gat(graph.x, graph.edge_index)
            preds = torch.softmax(logits, dim=1)[:, 1]
            test_preds.append(preds.cpu())
            test_labels.append(graph.y.cpu())
    
    test_preds = torch.cat(test_preds).numpy()
    test_labels = torch.cat(test_labels).numpy()
    test_auc = roc_auc_score(test_labels, test_preds)
    test_mcc = matthews_corrcoef(test_labels, (test_preds > 0.5).astype(int))
    
    print(f"[495 Test] AUC: {test_auc:.4f} | MCC: {test_mcc:.4f}")
    
    # 保存
    torch.save({
        'generator': generator.state_dict(),
        'edge_predictor': edge_predictor.state_dict(),
        'gat': gat.state_dict()
    }, "get_pre_result/data/wgan_311_demo1.pt")
    print("\n模型已保存")


if __name__ == "__main__":
    main()
