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


# ===================================================== 1. WGAN-GP (只用标签作为条件) =====================================================
class WGANGenerator(nn.Module):
    def __init__(self, noise_dim: int = 64, label_dim: int = 2, feat_dim: int = 1408, hidden_dim: int = 512):
        super().__init__()
        input_dim = noise_dim + label_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, feat_dim)
        )

    def forward(self, z: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.backbone(torch.cat([z, label], dim=-1))


class WGANDiscriminator(nn.Module):
    def __init__(self, feat_dim: int = 1408, label_dim: int = 2, hidden_dim: int = 512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(feat_dim + label_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.backbone(torch.cat([feat, label], dim=-1))


def gradient_penalty(discriminator, real_x: torch.Tensor, fake_x: torch.Tensor, label: torch.Tensor, device: torch.device) -> torch.Tensor:
    alpha = torch.rand(real_x.size(0), 1, device=device)
    interpolated = alpha * real_x + (1 - alpha) * fake_x
    interpolated.requires_grad_(True)
    d_interpolated = discriminator(interpolated, label)
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


# ===================================================== 4. 普通BCE Loss (不再使用Focal Loss) =====================================================
class BCELoss(nn.Module):
    """普通BCE Loss，数据已平衡后使用"""
    def __init__(self):
        super().__init__()
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [N, 2], labels: [N]
        loss = F.cross_entropy(logits, labels)
        return loss


# ===================================================== 5. 预训练 =====================================================
_global_discriminator = None


def pretrain_gan_and_edge(
    train_graphs: List[Data],
    device: torch.device,
    gan_epochs: int = 100,
    edge_epochs: int = 50,
    noise_dim: int = 64,
    label_dim: int = 2,
    lr: float = 1e-4,
    gan_batch_size: int = 32,
    lambda_gp: float = 10.0
) -> Tuple[WGANGenerator, EdgePredictor]:
    
    pos_feats = []
    for graph in train_graphs:
        graph = graph.to(device)
        pos_mask = graph.y == 1
        pos_idx = torch.where(pos_mask)[0]
        if len(pos_idx) >= 2:
            pos_feats.append(graph.x[pos_idx])
    
    pos_feats = torch.cat(pos_feats, dim=0)
    feat_dim = pos_feats.size(1)
    
    generator = WGANGenerator(noise_dim, label_dim, feat_dim).to(device)
    discriminator = WGANDiscriminator(feat_dim, label_dim).to(device)
    opt_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
    
    print("\n===== 预训练WGAN =====")
    n_samples = pos_feats.size(0)
    steps_per_epoch = max(1, n_samples // gan_batch_size)
    
    for epoch in range(gan_epochs):
        perm = torch.randperm(n_samples)
        for step in range(steps_per_epoch):
            idx = perm[step * gan_batch_size:(step + 1) * gan_batch_size]
            real_x = pos_feats[idx].to(device)
            
            batch_label = torch.zeros(len(idx), label_dim, device=device)
            batch_label[:, 1] = 1
            
            for _ in range(5):
                z = torch.randn(real_x.size(0), noise_dim, device=device)
                fake_x = generator(z, batch_label).detach()
                d_real = discriminator(real_x, batch_label).mean()
                d_fake = discriminator(fake_x, batch_label).mean()
                gp = gradient_penalty(discriminator, real_x, fake_x, batch_label, device)
                loss_D = d_fake - d_real + lambda_gp * gp
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()
            
            z = torch.randn(real_x.size(0), noise_dim, device=device)
            fake_x = generator(z, batch_label)
            loss_G = -discriminator(fake_x, batch_label).mean()
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
        
        if epoch % 20 == 0:
            print(f"[WGAN] Epoch {epoch:03d} | D: {loss_D.item():.4f} | G: {loss_G.item():.4f}")
    
    print("\n===== 预训练Edge Predictor =====")
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
    
    global _global_discriminator
    _global_discriminator = discriminator
    
    return generator, edge_predictor


# ===================================================== 6. 数据增强 =====================================================
def augment_dataset(
    train_graphs: List[Data],
    generator: WGANGenerator,
    edge_predictor: EdgePredictor,
    device: torch.device,
    label_dim: int = 2,
    target_ratio: float = 0.5,
    noise_dim: int = 64,
    edge_thresh: float = 0.5,
    max_new_nodes: int = 100
) -> List[Data]:
    
    augmented = []
    
    for graph in train_graphs:
        graph = deepcopy(graph).to(device)
        pos_mask = graph.y == 1
        n_pos = pos_mask.sum().item()
        n_neg = (graph.y == 0).sum().item()
        
        if n_pos == 0 or n_neg == 0:
            augmented.append(graph)
            continue
        
        target_pos = int(n_neg * target_ratio)
        n_need = max(0, min(target_pos - n_pos, max_new_nodes))
        
        if n_need == 0:
            augmented.append(graph)
            continue
        
        batch_label = torch.zeros(n_need, label_dim, device=device)
        batch_label[:, 1] = 1
        z = torch.randn(n_need, noise_dim, device=device)
        fake_feats = generator(z, batch_label)
        
        old_n_nodes = graph.x.size(0)
        graph.x = torch.cat([graph.x, fake_feats], dim=0)
        struct_dim = graph.struct_feat.size(1)
        graph.struct_feat = torch.cat([graph.struct_feat, fake_feats[:, :struct_dim]], dim=0)
        graph.seq_feat = torch.cat([graph.seq_feat, fake_feats[:, struct_dim:]], dim=0)
        graph.y = torch.cat([graph.y, torch.ones(n_need, device=device, dtype=graph.y.dtype)], dim=0)
        
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
        
        augmented.append(graph)
    
    return augmented


# ===================================================== 7. 训练GNN (使用普通BCE Loss) =====================================================
def train_gnn_only(
    train_graphs: List[Data],
    val_graphs: List[Data],
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3
) -> GATClassifier:
    
    gat = GATClassifier(in_dim=1408, hidden_dim=256, out_dim=2, heads=4).to(device)
    opt_gat = torch.optim.Adam(gat.parameters(), lr=lr)
    criterion = BCELoss()  # 使用普通BCE Loss
    
    best_val_mcc = -float('inf')
    best_state = None
    
    print("\n===== 训练GNN (使用普通BCE Loss) =====")
    for epoch in range(epochs):
        gat.train()
        total_loss = 0.0
        
        for graph in train_graphs:
            graph = graph.to(device)
            opt_gat.zero_grad()
            logits = gat(graph.x, graph.edge_index)
            loss = criterion(logits, graph.y)
            loss.backward()
            opt_gat.step()
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
            best_state = deepcopy(gat.state_dict())
        
        print(f"[Epoch {epoch:02d}] Loss: {total_loss/len(train_graphs):.4f} | Val AUC: {val_auc:.4f} | Val MCC: {val_mcc:.4f}")
    
    gat.load_state_dict(best_state)
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
    
    label_dim = 2
    
    # 预训练
    generator, edge_predictor = pretrain_gan_and_edge(
        train_graphs=train_graphs,
        device=device,
        gan_epochs=100,
        edge_epochs=50,
        noise_dim=64,
        label_dim=label_dim,
        lr=1e-4,
        gan_batch_size=32
    )
    
    # 预先生成增强数据
    print("\n===== 预先生成增强数据 =====")
    aug_train_graphs = augment_dataset(
        train_graphs=train_graphs,
        generator=generator,
        edge_predictor=edge_predictor,
        device=device,
        label_dim=label_dim,
        target_ratio=0.5
    )
    print(f"增强后训练集大小: {len(aug_train_graphs)}")
    
    # 训练GNN (使用普通BCE Loss)
    gat = train_gnn_only(
        train_graphs=aug_train_graphs,
        val_graphs=val_graphs,
        device=device,
        epochs=50,
        lr=1e-3
    )
    
    # 测试
    print("\n===== 测试结果 (495内部) =====")
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
    
    torch.save({
        'generator': generator.state_dict(),
        'edge_predictor': edge_predictor.state_dict(),
        'gat': gat.state_dict()
    }, "get_pre_result/data/wgan_312_demo3.pt")
    print("\n模型已保存")


if __name__ == "__main__":
    main()
