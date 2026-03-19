import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import negative_sampling, add_self_loops
from copy import deepcopy
from typing import List, Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score
import numpy as np


# ============================== 论文核心模块 1: 条件GAN (CGAN) 定向合成结合残基 ==============================
class CGANGenerator(nn.Module):
    """
    论文核心：条件GAN生成器 - 基于正类原型条件合成结合残基节点特征
    输入：随机噪声 + 正类结构原型
    输出：合成的节点特征（struct+seq拼接）
    """
    def __init__(self, noise_dim: int = 64, cond_dim: int = 128, feat_dim: int = 1408, hidden_dim: int = 512):
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

    def forward(self, z: torch.Tensor, cond_proto: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: 随机噪声 [B, noise_dim]
            cond_proto: 正类原型 [B, cond_dim]
        Returns:
            fake_feat: 合成节点特征 [B, feat_dim]
        """
        return self.backbone(torch.cat([z, cond_proto], dim=-1))


class CGANDiscriminator(nn.Module):
    """
    论文核心：条件GAN判别器 - 判别节点特征真实性（带条件）
    """
    def __init__(self, feat_dim: int = 1408, cond_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(feat_dim + cond_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, feat: torch.Tensor, cond_proto: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: 节点特征 [B, feat_dim]
            cond_proto: 正类原型 [B, cond_dim]
        Returns:
            score: 真实性分数 [B, 1]
        """
        return self.backbone(torch.cat([feat, cond_proto], dim=-1))


# ============================== 论文核心模块 2: 边缘生成器 (Edge Predictor) ==============================
class EdgePredictor(nn.Module):
    """
    论文核心：边缘生成器 - 学习蛋白质图拓扑连接规律
    输入：两个节点的特征对
    输出：边存在概率
    """
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
        """
        Args:
            h_i: 节点i特征 [B, node_dim]
            h_j: 节点j特征 [B, node_dim]
        Returns:
            prob: 边存在概率 [B]
        """
        return self.backbone(torch.cat([h_i, h_j], dim=-1)).squeeze(-1)


# ============================== 论文核心模块 3: GAT 分类器 (特征提取+残基预测) ==============================
class GATClassifier(nn.Module):
    """
    论文核心：GAT分类器 - 多注意力头提取氨基酸节点特征
    输入：节点特征 + 边索引
    输出：节点级二分类概率
    """
    def __init__(self, in_dim: int = 1408, hidden_dim: int = 256, out_dim: int = 2, heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        # 论文中8头注意力
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 节点特征 [N, in_dim]
            edge_index: 边索引 [2, E]
        Returns:
            logits: 分类logits [N, out_dim]
        """
        # 第一层GAT（多注意力头）
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        # 第二层GAT（单注意力头）
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        # 分类头
        logits = self.classifier(x)
        return logits


# ============================== 工具函数: Focal Loss (适配类不平衡) ==============================
class FocalLoss(nn.Module):
    """
    论文适配：Focal Loss替代普通交叉熵，更适合结合残基的极不平衡场景
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ============================== 论文流程 1: 预训练阶段 (GAN + 边缘生成器) ==============================
def pretrain_gan_and_edge(
    train_graphs: List[Data],
    device: torch.device,
    gan_epochs: int = 200,
    edge_epochs: int = 50,
    noise_dim: int = 64,
    lr: float = 1e-4
) -> Tuple[CGANGenerator, EdgePredictor]:
    """
    论文预训练步骤：先训练GAN生成高质量结合残基，再训练边缘生成器学习拓扑
    """
    # 1. 收集正类特征
    pos_feats = []
    pos_struct_feats = []
    for graph in train_graphs:
        graph = graph.to(device)
        pos_mask = graph.y == 1
        pos_idx = torch.where(pos_mask)[0]
        if len(pos_idx) < 2:
            continue
        pos_feats.append(graph.x[pos_idx])
        pos_struct_feats.append(graph.struct_feat[pos_idx])
    
    pos_feats = torch.cat(pos_feats, dim=0)
    pos_struct_feats = torch.cat(pos_struct_feats, dim=0)
    cond_proto = pos_struct_feats.mean(dim=0, keepdim=True)  # 正类原型
    cond_dim = cond_proto.size(1)
    feat_dim = pos_feats.size(1)

    # 2. 初始化CGAN
    generator = CGANGenerator(noise_dim, cond_dim, feat_dim).to(device)
    discriminator = CGANDiscriminator(feat_dim, cond_dim).to(device)
    opt_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    # 3. 训练CGAN
    print("\n===== 预训练阶段 1: 训练条件GAN =====")
    for epoch in range(gan_epochs):
        # 训练判别器
        real_x = pos_feats[torch.randperm(pos_feats.size(0))[:64]]
        batch_cond = cond_proto.repeat(real_x.size(0), 1)
        
        # 真实样本
        d_real = discriminator(real_x, batch_cond).mean()
        # 生成样本
        z = torch.randn(real_x.size(0), noise_dim, device=device)
        fake_x = generator(z, batch_cond)
        d_fake = discriminator(fake_x.detach(), batch_cond).mean()
        
        # 判别器损失（论文公式：L_d = -E[D(real)] - E[1-D(fake)]）
        loss_D = -(torch.log(d_real + 1e-8) + torch.log(1 - d_fake + 1e-8)).mean()
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # 训练生成器
        z = torch.randn(real_x.size(0), noise_dim, device=device)
        fake_x = generator(z, batch_cond)
        d_fake = discriminator(fake_x, batch_cond).mean()
        # 生成器损失（论文公式：L_g = -E[D(fake)]）
        loss_G = -torch.log(d_fake + 1e-8).mean()
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        if epoch % 50 == 0:
            print(f"[CGAN] Epoch {epoch:03d} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

    # 4. 初始化并训练边缘生成器
    print("\n===== 预训练阶段 2: 训练边缘生成器 =====")
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
            
            # 负采样
            neg_edges = negative_sampling(pos_edges, num_nodes=node_feat.size(0), num_neg_samples=pos_edges.size(1))
            
            # 边损失（论文公式：L_edge = BCE(real) + BCE(fake)）
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
            print(f"[Edge Predictor] Epoch {epoch:03d} | Loss: {total_loss:.4f}")

    # 冻结预训练模型参数
    generator.eval()
    edge_predictor.eval()
    for p in generator.parameters():
        p.requires_grad = False
    for p in edge_predictor.parameters():
        p.requires_grad = False

    return generator, edge_predictor


# ============================== 论文流程 2: 图增强 (GAN合成节点 + 边缘生成器连边) ==============================
def augment_graph_with_gnngan(
    graph: Data,
    generator: CGANGenerator,
    edge_predictor: EdgePredictor,
    device: torch.device,
    target_ratio: float = 0.5,  # 论文建议合成后结合残基占比30%-50%
    noise_dim: int = 64,
    edge_thresh: float = 0.5  # 论文边阈值T=0.5
) -> Data:
    """
    论文数据融合策略：合成结合残基节点 + 边缘生成器预测边 + 构建平衡图
    """
    graph = deepcopy(graph).to(device)
    pos_mask = graph.y == 1
    neg_mask = graph.y == 0
    n_pos = pos_mask.sum().item()
    n_neg = neg_mask.sum().item()

    if n_pos == 0 or n_neg == 0:
        return graph

    # 论文类平衡采样规则：计算需要合成的结合残基数量
    target_pos = int(n_neg * target_ratio)
    n_need = max(0, target_pos - n_pos)
    if n_need == 0:
        return graph

    # 1. 合成结合残基节点特征
    pos_struct_feat = graph.struct_feat[pos_mask]
    cond_proto = pos_struct_feat.mean(dim=0, keepdim=True)
    z = torch.randn(n_need, noise_dim, device=device)
    fake_feats = generator(z, cond_proto.repeat(n_need, 1))

    # 2. 扩展图结构
    old_n_nodes = graph.x.size(0)
    graph.x = torch.cat([graph.x, fake_feats], dim=0)
    struct_dim = graph.struct_feat.size(1)
    graph.struct_feat = torch.cat([graph.struct_feat, fake_feats[:, :struct_dim]], dim=0)
    graph.seq_feat = torch.cat([graph.seq_feat, fake_feats[:, struct_dim:]], dim=0)
    graph.y = torch.cat([graph.y, torch.ones(n_need, device=device, dtype=graph.y.dtype)], dim=0)

    # 3. 边缘生成器预测边（论文公式7：概率>阈值则连边）
    new_edges = []
    for i in range(n_need):
        new_id = old_n_nodes + i
        new_feat = graph.x[new_id:new_id+1]
        old_feats = graph.x[:old_n_nodes]
        
        # 预测新节点与所有旧节点的连边概率
        probs = edge_predictor(new_feat.repeat(old_n_nodes, 1), old_feats)
        # 阈值筛选
        connect_mask = probs > edge_thresh
        connect_idx = torch.where(connect_mask)[0]
        
        # 构建无向边
        for j in connect_idx:
            new_edges.append([new_id, j.item()])
            new_edges.append([j.item(), new_id])

    # 4. 拼接边
    if new_edges:
        edge_add = torch.tensor(new_edges, device=device, dtype=torch.long).t()
        graph.edge_index = torch.cat([graph.edge_index, edge_add], dim=1)

    return graph


# ============================== 论文流程 3: 分阶段同步训练 (GAN+GNN联合优化) ==============================
def train_gnngan_full(
    train_graphs: List[Data],
    val_graphs: List[Data],
    generator: CGANGenerator,
    edge_predictor: EdgePredictor,
    device: torch.device,
    gat_epochs: int = 50,
    lr: float = 1e-3,
    lambda_gan: float = 0.1,  # GAN损失权重
    lambda_edge: float = 0.1  # 边损失权重
) -> GATClassifier:
    """
    论文分阶段同步训练：阶段2联合训练GAT+微调GAN+边缘生成器
    """
    # 初始化GAT分类器
    gat = GATClassifier(in_dim=1408, hidden_dim=256, out_dim=2, heads=4).to(device)
    opt_gat = torch.optim.Adam(gat.parameters(), lr=lr)
    focal_loss = FocalLoss(alpha=0.75, gamma=2.0)

    # 解冻部分参数用于微调
    for p in generator.parameters():
        p.requires_grad = True
    for p in edge_predictor.parameters():
        p.requires_grad = True
    opt_G = torch.optim.Adam(generator.parameters(), lr=1e-5)  # 微调GAN用小学习率
    opt_edge = torch.optim.Adam(edge_predictor.parameters(), lr=1e-5)

    best_val_mcc = -float('inf')
    best_gat_weights = None

    print("\n===== 同步训练阶段: GAN+GNN联合优化 =====")
    for epoch in range(gat_epochs):
        gat.train()
        generator.train()
        edge_predictor.train()
        total_loss = 0.0

        for graph in train_graphs:
            graph = graph.to(device)
            # 1. 图增强（每轮动态增强）
            aug_graph = augment_graph_with_gnngan(graph, generator, edge_predictor, device, target_ratio=0.4)
            
            # 2. GAT分类
            logits = gat(aug_graph.x, aug_graph.edge_index)
            # 分类损失（论文L_node，用Focal Loss适配）
            loss_node = focal_loss(logits, aug_graph.y)

            # 3. GAN损失（微调生成器）
            pos_mask = aug_graph.y == 1
            pos_idx = torch.where(pos_mask)[0]
            if len(pos_idx) > 0:
                real_pos_feat = aug_graph.x[pos_idx]
                cond_proto = aug_graph.struct_feat[pos_idx].mean(dim=0, keepdim=True)
                z = torch.randn(len(pos_idx), 64, device=device)
                fake_pos_feat = generator(z, cond_proto.repeat(len(pos_idx), 1))
                # 生成器损失
                loss_gan = F.mse_loss(fake_pos_feat, real_pos_feat)
            else:
                loss_gan = torch.tensor(0.0, device=device)

            # 4. 边损失（微调边缘生成器）
            pos_edges = aug_graph.edge_index
            neg_edges = negative_sampling(pos_edges, num_nodes=aug_graph.x.size(0), num_neg_samples=pos_edges.size(1))
            def edge_loss(edges, label):
                i, j = edges
                probs = edge_predictor(aug_graph.x[i], aug_graph.x[j])
                return F.binary_cross_entropy(probs, torch.full_like(probs, label))
            loss_edge = edge_loss(pos_edges, 1.0) + edge_loss(neg_edges, 0.0)

            # 5. 总损失（论文公式15：L_total = L_node + λ_gan*L_gan + λ_edge*L_edge）
            loss = loss_node + lambda_gan * loss_gan + lambda_edge * loss_edge

            # 反向传播
            opt_gat.zero_grad()
            opt_G.zero_grad()
            opt_edge.zero_grad()
            loss.backward()
            opt_gat.step()
            opt_G.step()
            opt_edge.step()

            total_loss += loss.item()

        # 验证集评估
        gat.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for graph in val_graphs:
                graph = graph.to(device)
                logits = gat(graph.x, graph.edge_index)
                preds = torch.softmax(logits, dim=1)[:, 1]
                val_preds.append(preds.cpu().numpy())
                val_labels.append(graph.y.cpu().numpy())
        
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_auc = roc_auc_score(val_labels, val_preds)
        val_ap = average_precision_score(val_labels, val_preds)
        val_mcc = matthews_corrcoef(val_labels, (val_preds > 0.5).astype(int))
        val_f1 = f1_score(val_labels, (val_preds > 0.5).astype(int))

        # 保存最佳模型
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_gat_weights = deepcopy(gat.state_dict())

        print(f"[Epoch {epoch:03d}] Total Loss: {total_loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f} | Val MCC: {val_mcc:.4f} | Val F1: {val_f1:.4f}")

    # 加载最佳模型
    if best_gat_weights is not None:
        gat.load_state_dict(best_gat_weights)
    
    return gat


# ============================== 主函数: 完整流程 ==============================
def main():
    # 1. 设备初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 加载数据（你的数据格式：struct_feat, seq_feat, x, edge_index, y）
    print("\n===== 加载数据 =====")
    graph_list = torch.load("get_graph_data/data/pyg_graph_datas_495_train.pt", weights_only=False)
    
    # 过滤有效图
    def is_valid(graph):
        pos_idx = torch.where(graph.y == 1)[0]
        return len(pos_idx) >= 2 and graph.edge_index.size(1) >= 2 and graph.x.size(0) >= 2
    graph_list = [g for g in graph_list if is_valid(g)]
    print(f"有效图数量: {len(graph_list)}")

    # 3. 数据划分
    random.shuffle(graph_list)
    n_total = len(graph_list)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    train_graphs = graph_list[:n_train]
    val_graphs = graph_list[n_train:n_train+n_val]
    test_graphs = graph_list[n_train+n_val:]
    print(f"数据划分: Train {len(train_graphs)} | Val {len(val_graphs)} | Test {len(test_graphs)}")

    # 4. 预训练GAN和边缘生成器
    generator, edge_predictor = pretrain_gan_and_edge(
        train_graphs=train_graphs,
        device=device,
        gan_epochs=200,
        edge_epochs=50,
        lr=1e-4
    )

    # 5. 分阶段同步训练GNN-GAN
    gat = train_gnngan_full(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        generator=generator,
        edge_predictor=edge_predictor,
        device=device,
        gat_epochs=100,
        lr=1e-3,
        lambda_gan=0.1,
        lambda_edge=0.1
    )

    # 6. 测试集评估
    print("\n===== 测试集评估 =====")
    gat.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for graph in test_graphs:
            graph = graph.to(device)
            logits = gat(graph.x, graph.edge_index)
            preds = torch.softmax(logits, dim=1)[:, 1]
            test_preds.append(preds.cpu().numpy())
            test_labels.append(graph.y.cpu().numpy())
    
    test_preds = np.concatenate(test_preds)
    test_labels = np.concatenate(test_labels)
    test_auc = roc_auc_score(test_labels, test_preds)
    test_ap = average_precision_score(test_labels, test_preds)
    test_mcc = matthews_corrcoef(test_labels, (test_preds > 0.5).astype(int))
    test_f1 = f1_score(test_labels, (test_preds > 0.5).astype(int))

    print(f"[Test Results] AUC: {test_auc:.4f} | AP: {test_ap:.4f} | MCC: {test_mcc:.4f} | F1: {test_f1:.4f}")

    # 7. 保存模型
    torch.save({
        'generator': generator.state_dict(),
        'edge_predictor': edge_predictor.state_dict(),
        'gat': gat.state_dict()
    }, "get_pre_result/data/gnngan_model.pt")
    print("\n===== 模型保存完成 =====")


if __name__ == "__main__":
    main()
