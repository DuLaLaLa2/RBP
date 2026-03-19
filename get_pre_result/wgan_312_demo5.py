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
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np


# ===================================================== 0. VGMM噪声生成器 =====================================================
class VGMMNoiseGenerator:
    def __init__(self, n_components: int = 5, n_pca: int = 64, random_state: int = 42):
        self.n_components = n_components
        self.n_pca = n_pca
        self.random_state = random_state
        self.vgmm = None
        self.scaler = StandardScaler()
        self.pca = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray) -> bool:
        if X.shape[0] < self.n_components:
            print(f"[VGMM] 样本数不足，跳过")
            return False
        
        X_scaled = self.scaler.fit_transform(X)
        
        if X_scaled.shape[1] > self.n_pca:
            self.pca = PCA(n_components=min(self.n_pca, X_scaled.shape[1]))
            X_processed = self.pca.fit_transform(X_scaled)
        else:
            X_processed = X_scaled
        
        self.vgmm = BayesianGaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            weight_concentration_prior_type='dirichlet_process',
            max_iter=200,
            random_state=self.random_state
        )
        self.vgmm.fit(X_processed)
        self.is_fitted = True
        
        print(f"[VGMM] 训练完成: {self.n_components}个成分, 维度{X_processed.shape[1]}")
        return True
    
    def sample(self, n: int) -> np.ndarray:
        if not self.is_fitted:
            return np.random.randn(n, self.n_pca)
        
        samples, _ = self.vgmm.sample(n)
        
        if samples.shape[1] < self.n_pca:
            padding = np.random.randn(n, self.n_pca - samples.shape[1])
            samples = np.hstack([samples, padding])
        elif samples.shape[1] > self.n_pca:
            samples = samples[:, :self.n_pca]
        
        return samples


# ===================================================== 1. 样本筛选器 (基于最近邻距离) =====================================================
class SampleFilter:
    """
    基于最近邻距离筛选合成样本
    - 下限阈值: 小于说明太相似，丢弃
    - 上限阈值: 大于说明离群点，丢弃
    - 区间内: 保留 (既有多样性又不至于太离谱)
    """
    def __init__(self, k_neighbors: int = 5, lower_ratio: float = 0.3, upper_ratio: float = 2.0):
        """
        Args:
            k_neighbors: 计算最近邻的k值
            lower_ratio: 下限比例 (相对于真实样本平均距离)
            upper_ratio: 上限比例 (相对于真实样本平均距离)
        """
        self.k = k_neighbors
        self.lower_ratio = lower_ratio
        self.upper_ratio = upper_ratio
        self.real_samples = None
        self.nbrs = None
        self.real_mean_dist = None
        
    def fit(self, real_samples: np.ndarray):
        """用真实样本构建最近邻模型"""
        self.real_samples = real_samples
        
        # 计算真实样本自身的平均最近邻距离
        if len(real_samples) > self.k:
            self.nbrs = NearestNeighbors(n_neighbors=self.k+1, algorithm='ball_tree').fit(real_samples)
            distances, _ = self.nbrs.kneighbors(real_samples)
            # 排除自身，距离从第2个开始
            self.real_mean_dist = distances[:, 1:].mean()
        else:
            self.real_mean_dist = 1.0
        
        print(f"[SampleFilter] 真实样本平均最近邻距离: {self.real_mean_dist:.4f}")
        
    def filter(self, synthetic_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        筛选合成样本
        Returns:
            (保留的样本, 保留样本的mask)
        """
        if self.real_samples is None or len(synthetic_samples) == 0:
            return synthetic_samples, np.ones(len(synthetic_samples), dtype=bool)
        
        # 计算每个合成样本到真实样本的最近邻距离
        distances, _ = self.nbrs.kneighbors(synthetic_samples)
        min_distances = distances[:, 0]  # 最近邻距离
        
        # 设置阈值
        lower_thresh = self.real_mean_dist * self.lower_ratio
        upper_thresh = self.real_mean_dist * self.upper_ratio
        
        # 筛选
        mask = (min_distances >= lower_thresh) & (min_distances <= upper_thresh)
        
        kept = synthetic_samples[mask]
        
        print(f"[SampleFilter] 合成样本: {len(synthetic_samples)}, 保留: {kept.shape[0]}, "
              f"太相似(丢弃): {(min_distances < lower_thresh).sum()}, "
              f"离群点(丢弃): {(min_distances > upper_thresh).sum()}")
        
        return kept, mask


# ===================================================== 2. WGAN-GP =====================================================
class WGANGenerator(nn.Module):
    def __init__(self, vgm_dim: int = 64, noise_dim: int = 32, label_dim: int = 2, 
                 feat_dim: int = 1408, hidden_dim: int = 512):
        super().__init__()
        input_dim = vgm_dim + noise_dim + label_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, feat_dim)
        )

    def forward(self, vgm_sample: torch.Tensor, noise: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.backbone(torch.cat([vgm_sample, noise, label], dim=-1))


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


# ===================================================== 3. 边缘预测器 =====================================================
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


# ===================================================== 4. GAT分类器 =====================================================
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


class BCELoss(nn.Module):
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)


# ===================================================== 5. 预训练 =====================================================
_global_discriminator = None


def pretrain_vgmm_gan_edge(
    train_graphs: List[Data],
    device: torch.device,
    vgm_n_components: int = 5,
    vgm_n_pca: int = 64,
    gan_epochs: int = 100,
    edge_epochs: int = 50,
    noise_dim: int = 32,
    label_dim: int = 2,
    lr: float = 1e-4,
    gan_batch_size: int = 32,
    lambda_gp: float = 10.0
) -> Tuple[WGANGenerator, EdgePredictor, VGMMNoiseGenerator]:
    
    pos_feats = []
    for graph in train_graphs:
        graph = graph.to(device)
        pos_mask = graph.y == 1
        pos_idx = torch.where(pos_mask)[0]
        if len(pos_idx) >= 2:
            pos_feats.append(graph.x[pos_idx].cpu().numpy())
    
    pos_feats = np.vstack(pos_feats)
    feat_dim = pos_feats.shape[1]
    print(f"[Data] 正类样本数: {pos_feats.shape[0]}, 维度: {pos_feats.shape[1]}")
    
    # VGMM
    print("\n===== 阶段1: 训练VGMM =====")
    vgm_generator = VGMMNoiseGenerator(n_components=vgm_n_components, n_pca=vgm_n_pca)
    vgm_generator.fit(pos_feats)
    vgm_dim = vgm_n_pca
    
    # WGAN
    print("\n===== 阶段2: 训练WGAN =====")
    generator = WGANGenerator(vgm_dim, noise_dim, label_dim, feat_dim).to(device)
    discriminator = WGANDiscriminator(feat_dim, label_dim).to(device)
    opt_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
    
    n_samples = pos_feats.shape[0]
    steps_per_epoch = max(1, n_samples // gan_batch_size)
    
    for epoch in range(gan_epochs):
        perm = np.random.permutation(n_samples)
        for step in range(steps_per_epoch):
            idx = perm[step * gan_batch_size:(step + 1) * gan_batch_size]
            real_x = torch.tensor(pos_feats[idx], dtype=torch.float32).to(device)
            batch_label = torch.zeros(len(idx), label_dim, device=device)
            batch_label[:, 1] = 1
            
            vgm_s = torch.tensor(vgm_generator.sample(len(idx)), dtype=torch.float32).to(device)
            noise = torch.randn(len(idx), noise_dim, device=device)
            
            for _ in range(5):
                z = torch.randn(real_x.size(0), noise_dim, device=device)
                fake_x = generator(vgm_s, z, batch_label).detach()
                d_real = discriminator(real_x, batch_label).mean()
                d_fake = discriminator(fake_x, batch_label).mean()
                gp = gradient_penalty(discriminator, real_x, fake_x, batch_label, device)
                loss_D = d_fake - d_real + lambda_gp * gp
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()
            
            z = torch.randn(real_x.size(0), noise_dim, device=device)
            fake_x = generator(vgm_s, z, batch_label)
            loss_G = -discriminator(fake_x, batch_label).mean()
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
        
        if epoch % 20 == 0:
            print(f"[WGAN] Epoch {epoch:03d} | D: {loss_D.item():.4f} | G: {loss_G.item():.4f}")
    
    # Edge
    print("\n===== 阶段3: 训练Edge =====")
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
    
    return generator, edge_predictor, vgm_generator, pos_feats


# ===================================================== 6. 数据增强 (含筛选) =====================================================
def augment_dataset(
    train_graphs: List[Data],
    generator: WGANGenerator,
    edge_predictor: EdgePredictor,
    vgm_generator: VGMMNoiseGenerator,
    sample_filter: SampleFilter,
    device: torch.device,
    noise_dim: int = 32,
    label_dim: int = 2,
    target_ratio: float = 0.5,
    edge_thresh: float = 0.5,
    max_new_nodes: int = 100,
    oversample_factor: float = 3.0  # 多生成一些用于筛选
) -> List[Data]:
    
    augmented = []
    vgm_dim = vgm_generator.n_pca
    
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
        
        # 多生成一些用于筛选
        n_generate = int(n_need * oversample_factor)
        
        # 生成
        vgm_s = torch.tensor(vgm_generator.sample(n_generate), dtype=torch.float32).to(device)
        noise = torch.randn(n_generate, noise_dim, device=device)
        batch_label = torch.zeros(n_generate, label_dim, device=device)
        batch_label[:, 1] = 1
        
        fake_feats = generator(vgm_s, noise, batch_label)
        fake_np = fake_feats.cpu().detach().numpy()
        
        # 筛选
        fake_np_filtered, mask = sample_filter.filter(fake_np)
        
        if len(fake_np_filtered) == 0:
            augmented.append(graph)
            continue
        
        # 只取需要的数量
        fake_np_filtered = fake_np_filtered[:n_need]
        
        # 扩展图
        fake_feats_filtered = torch.tensor(fake_np_filtered, dtype=torch.float32).to(device)
        
        old_n_nodes = graph.x.size(0)
        graph.x = torch.cat([graph.x, fake_feats_filtered], dim=0)
        struct_dim = graph.struct_feat.size(1)
        graph.struct_feat = torch.cat([graph.struct_feat, fake_feats_filtered[:, :struct_dim]], dim=0)
        graph.seq_feat = torch.cat([graph.seq_feat, fake_feats_filtered[:, struct_dim:]], dim=0)
        graph.y = torch.cat([graph.y, torch.ones(len(fake_np_filtered), device=device, dtype=graph.y.dtype)], dim=0)
        
        # 边
        if old_n_nodes > 0:
            old_nodes = graph.x[:old_n_nodes]
            new_edges = []
            for new_idx in range(len(fake_np_filtered)):
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


# ===================================================== 7. 训练GNN =====================================================
def train_gnn_only(
    train_graphs: List[Data],
    val_graphs: List[Data],
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3
) -> GATClassifier:
    
    gat = GATClassifier(in_dim=1408, hidden_dim=256, out_dim=2, heads=4).to(device)
    opt_gat = torch.optim.Adam(gat.parameters(), lr=lr)
    criterion = BCELoss()
    
    best_val_mcc = -float('inf')
    best_state = None
    
    print("\n===== 训练GNN =====")
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
    
    # 预训练
    generator, edge_predictor, vgm_generator, pos_feats = pretrain_vgmm_gan_edge(
        train_graphs=train_graphs,
        device=device,
        vgm_n_components=5,
        vgm_n_pca=64,
        gan_epochs=100,
        edge_epochs=50,
        noise_dim=32,
        label_dim=2,
        lr=1e-4,
        gan_batch_size=32
    )
    
    # 构建样本筛选器
    print("\n===== 构建样本筛选器 =====")
    sample_filter = SampleFilter(k_neighbors=5, lower_ratio=0.3, upper_ratio=2.0)
    sample_filter.fit(pos_feats)
    
    # 增强数据
    print("\n===== 预先生成增强数据 =====")
    aug_train_graphs = augment_dataset(
        train_graphs=train_graphs,
        generator=generator,
        edge_predictor=edge_predictor,
        vgm_generator=vgm_generator,
        sample_filter=sample_filter,
        device=device,
        target_ratio=0.5,
        oversample_factor=2.0
    )
    print(f"增强后训练集大小: {len(aug_train_graphs)}")
    
    # 训练GNN
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
    }, "get_pre_result/data/wgan_312_demo5.pt")
    print("\n模型已保存")


if __name__ == "__main__":
    main()
