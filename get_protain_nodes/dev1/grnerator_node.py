"""
蛋白质图增强系统 - 使用Transformer GAN生成合成节点
输入：495个蛋白质图的.pt文件
输出：增强后的495个蛋白质图保存到一个.pt文件中
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from tqdm import tqdm

# ==================== 1. 工具函数 ====================

def set_seed(seed=42):
    """设置随机种子以保证可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==================== 2. 位置编码模块 ====================

class PositionalEncoding(nn.Module):
    """
    位置编码 - 为Transformer提供序列位置信息
    
    Args:
        d_model: 模型维度
        dropout: dropout率
        max_len: 最大序列长度
    
    Input:
        x: [batch_size, seq_len, d_model]
    
    Output:
        [batch_size, seq_len, d_model] 添加位置编码后的张量
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算除数项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        # 偶数列使用sin，奇数列使用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ==================== 3. Transformer生成器 ====================

class TransformerProteinGenerator(nn.Module):
    """
    基于Transformer的蛋白质节点生成器
    
    功能：从随机噪声生成新的蛋白质节点特征
    
    Args:
        node_feature_dim: 节点特征维度（如ESM模型输出，通常1280）
        latent_dim: 潜在空间维度（随机噪声维度）
        n_classes: 蛋白质类别数
        d_model: Transformer模型维度
        nhead: 注意力头数
        num_layers: Transformer层数
        dim_feedforward: FFN维度
        dropout: dropout率
    
    Input:
        noise: [batch_size, latent_dim] 随机噪声
        class_cond: [batch_size, n_classes] 类别条件（one-hot）
        context: [batch_size, context_dim] 图上下文特征
        n_generate: 要生成的节点数量
    
    Output:
        generated_nodes: [batch_size, n_generate, node_feature_dim]
    """
    def __init__(self, 
                 node_feature_dim: int = 1280,
                 latent_dim: int = 100,
                 n_classes: int = 10,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.node_feature_dim = node_feature_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        
        # 输入投影层
        self.noise_proj = nn.Linear(latent_dim, d_model)
        self.class_proj = nn.Linear(n_classes, d_model)
        self.context_proj = nn.Linear(128, d_model)  # 上下文特征维度为128
        
        # 位置编码
        self.position_encoding = PositionalEncoding(d_model, dropout, max_len=100)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, node_feature_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, 
                noise: torch.Tensor, 
                class_cond: torch.Tensor, 
                context: torch.Tensor,
                n_generate: int = 1) -> torch.Tensor:
        
        batch_size = noise.size(0)
        
        # 1. 投影所有输入到d_model维度
        noise_feat = self.noise_proj(noise)  # [batch_size, d_model]
        class_feat = self.class_proj(class_cond)  # [batch_size, d_model]
        context_feat = self.context_proj(context)  # [batch_size, d_model]
        
        # 2. 融合特征（相加）
        fused_feat = noise_feat + class_feat + context_feat
        
        # 3. 扩展为序列
        fused_feat = fused_feat.unsqueeze(1).repeat(1, n_generate, 1)  # [batch, n_generate, d_model]
        
        # 4. 添加位置编码
        fused_feat = self.position_encoding(fused_feat)
        
        # 5. Transformer编码
        transformer_output = self.transformer_encoder(fused_feat)  # [batch, n_generate, d_model]
        
        # 6. 输出投影
        generated_nodes = self.output_proj(transformer_output)  # [batch, n_generate, node_feature_dim]
        
        return generated_nodes

# ==================== 4. Transformer判别器 ====================

class TransformerProteinDiscriminator(nn.Module):
    """
    基于Transformer的蛋白质节点判别器
    
    功能：判断节点是真实的还是生成的
    
    Args:
        node_feature_dim: 节点特征维度
        d_model: Transformer模型维度
        nhead: 注意力头数
        num_layers: Transformer层数
    
    Input:
        node_features: [batch_size, num_nodes, node_feature_dim] 节点特征
    
    Output:
        [batch_size, 1] 真实性概率（0-1之间）
    """
    def __init__(self, 
                 node_feature_dim: int = 1280,
                 d_model: int = 256,
                 nhead: int = 4,
                 num_layers: int = 3):
        super().__init__()
        
        # 节点特征投影
        self.input_proj = nn.Linear(node_feature_dim, d_model)
        
        # 位置编码
        self.position_encoding = PositionalEncoding(d_model, dropout=0.1, max_len=200)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        # 1. 投影
        x = self.input_proj(node_features)  # [batch, num_nodes, d_model]
        
        # 2. 添加位置编码
        x = self.position_encoding(x)
        
        # 3. Transformer编码
        x = self.transformer_encoder(x)  # [batch, num_nodes, d_model]
        
        # 4. 全局池化（取所有节点的平均值）
        x = x.mean(dim=1)  # [batch, d_model]
        
        # 5. 分类
        out = self.classifier(x)  # [batch, 1]
        
        return out

# ==================== 5. 图上下文提取器 ====================

class GraphContextExtractor(nn.Module):
    """
    从蛋白质图中提取全局上下文特征
    
    功能：将整个蛋白质图编码为一个固定长度的向量
    
    Args:
        node_feature_dim: 节点特征维度
        context_dim: 输出上下文特征维度
    
    Input:
        protein_graph: 包含 .x (节点特征) 和 .edge_index 的图对象
    
    Output:
        context: [1, context_dim] 图上下文特征
    """
    def __init__(self, node_feature_dim: int = 1280, context_dim: int = 128):
        super().__init__()
        
        # 全局池化层
        self.global_pool = nn.Sequential(
            nn.Linear(node_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, context_dim)
        )
        
        # 统计特征编码器
        self.stat_encoder = nn.Sequential(
            nn.Linear(5, 64),  # 5个统计量：mean, std, max, min, avg_degree
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 融合层
        self.fusion = nn.Linear(context_dim + 32, context_dim)
    
    def compute_avg_degree(self, graph) -> float:
        """计算图的平均度"""
        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
            num_edges = graph.edge_index.size(1)
            num_nodes = graph.num_nodes
            return (2 * num_edges) / num_nodes if num_nodes > 0 else 0
        return 0
    
    def forward(self, protein_graph) -> torch.Tensor:
        node_features = protein_graph.x  # [num_nodes, node_dim]
        
        # 1. 全局池化
        global_feat = node_features.mean(dim=0, keepdim=True)  # [1, node_dim]
        global_feat = self.global_pool(global_feat)  # [1, context_dim]
        
        # 2. 统计特征
        stats = torch.tensor([
            node_features.mean().item(),
            node_features.std().item(),
            node_features.max().item(),
            node_features.min().item(),
            self.compute_avg_degree(protein_graph)
        ]).to(node_features.device)
        
        stats_feat = self.stat_encoder(stats.unsqueeze(0))  # [1, 32]
        
        # 3. 融合
        context = self.fusion(torch.cat([global_feat, stats_feat], dim=1))  # [1, context_dim]
        
        return context

# ==================== 6. 图数据集 ====================

class ProteinGraphDataset(Dataset):
    """
    蛋白质图数据集
    
    功能：加载所有蛋白质图，为训练提供数据
    
    Args:
        graph_path: .pt文件路径，包含所有蛋白质图
        device: 设备
    
    Returns:
        __getitem__返回单个蛋白质图
    """
    def __init__(self, graph_path: str, device: str = 'cuda'):
        self.graphs = torch.load(graph_path)
        self.device = device
        print(f"加载了 {len(self.graphs)} 个蛋白质图")
        
        # 检查每个图的属性
        for i, graph in enumerate(self.graphs):
            if not hasattr(graph, 'node_labels'):
                # 如果没有标签，创建默认标签（假设所有节点属于同一类）
                graph.node_labels = torch.zeros(graph.num_nodes, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int):
        return self.graphs[idx]

# ==================== 7. 合成节点连接器 ====================

class SyntheticNodeConnector:
    """
    合成节点连接器 - 确定新节点之间的连接关系
    
    功能：根据选择的策略，决定合成节点之间以及合成节点与原图节点的连接
    
    Args:
        strategy: 连接策略
            - 'independent': 合成节点之间不连接
            - 'cluster': 形成新集群（基于相似度）
            - 'chain': 形成链状结构
            - 'fully_connected': 全连接
            - 'data_driven': 基于原图模式学习
        similarity_threshold: 相似度阈值（用于cluster策略）
    
    Input:
        original_graph: 原图
        synthetic_features: [n_syn, feat_dim] 合成节点特征
        synthetic_info: 每个合成节点的生成信息（包含锚点等）
    
    Output:
        edges: 新边的列表，每个元素为 [u, v]
    """
    def __init__(self, strategy: str = 'data_driven', similarity_threshold: float = 0.7):
        self.strategy = strategy
        self.similarity_threshold = similarity_threshold
    
    def compute_similarity(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """计算两个节点特征的余弦相似度"""
        return F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()
    
    def are_nodes_connected(self, graph, node1: int, node2: int) -> bool:
        """检查原图中两个节点是否相连"""
        if not hasattr(graph, 'edge_index') or graph.edge_index is None:
            return False
        
        # 检查是否存在边
        edges = graph.edge_index
        for i in range(edges.size(1)):
            if (edges[0, i] == node1 and edges[1, i] == node2) or \
               (edges[0, i] == node2 and edges[1, i] == node1):
                return True
        return False
    
    def find_common_neighbors(self, graph, node1: int, node2: int) -> List[int]:
        """找到两个节点的共同邻居"""
        if not hasattr(graph, 'edge_index') or graph.edge_index is None:
            return []
        
        # 获取每个节点的邻居
        neighbors1 = set()
        neighbors2 = set()
        edges = graph.edge_index
        
        for i in range(edges.size(1)):
            if edges[0, i] == node1:
                neighbors1.add(edges[1, i].item())
            if edges[1, i] == node1:
                neighbors1.add(edges[0, i].item())
            if edges[0, i] == node2:
                neighbors2.add(edges[1, i].item())
            if edges[1, i] == node2:
                neighbors2.add(edges[0, i].item())
        
        return list(neighbors1 & neighbors2)
    
    def connect_to_original(self, 
                           original_graph, 
                           synthetic_features: torch.Tensor,
                           synthetic_info: List[Dict]) -> List[List[int]]:
        """
        确定合成节点与原图节点的连接
        
        基于特征相似度连接
        """
        n_original = original_graph.num_nodes
        n_synthetic = len(synthetic_features)
        edges = []
        
        for i in range(n_synthetic):
            # 计算与所有原节点的相似度
            similarities = []
            for j in range(n_original):
                sim = self.compute_similarity(
                    synthetic_features[i], 
                    original_graph.x[j]
                )
                similarities.append((j, sim))
            
            # 按相似度排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 连接相似度最高的2-3个节点
            n_connections = min(3, len(similarities))
            for k in range(n_connections):
                if similarities[k][1] > 0.5:  # 相似度阈值
                    orig_node = similarities[k][0]
                    syn_node = n_original + i
                    edges.append([syn_node, orig_node])
                    edges.append([orig_node, syn_node])
        
        return edges
    
    def connect_among_synthetic(self,
                               synthetic_features: torch.Tensor,
                               synthetic_info: List[Dict],
                               original_graph) -> List[List[int]]:
        """
        确定合成节点之间的连接
        """
        n_syn = len(synthetic_features)
        n_original = original_graph.num_nodes
        edges = []
        
        if n_syn <= 1:
            return edges
        
        if self.strategy == 'independent':
            # 策略1：不连接
            pass
            
        elif self.strategy == 'cluster':
            # 策略2：基于相似度形成集群
            for i in range(n_syn):
                for j in range(i+1, n_syn):
                    similarity = self.compute_similarity(
                        synthetic_features[i], 
                        synthetic_features[j]
                    )
                    if similarity > self.similarity_threshold:
                        syn_i = n_original + i
                        syn_j = n_original + j
                        edges.append([syn_i, syn_j])
                        edges.append([syn_j, syn_i])
            
        elif self.strategy == 'chain':
            # 策略3：形成链状
            for i in range(n_syn - 1):
                syn_i = n_original + i
                syn_j = n_original + i + 1
                edges.append([syn_i, syn_j])
                edges.append([syn_j, syn_i])
            
        elif self.strategy == 'fully_connected':
            # 策略4：全连接
            for i in range(n_syn):
                for j in range(n_syn):
                    if i != j:
                        syn_i = n_original + i
                        syn_j = n_original + j
                        edges.append([syn_i, syn_j])
            
        elif self.strategy == 'data_driven':
            # 策略5：基于锚点关系
            for i, info_i in enumerate(synthetic_info):
                for j, info_j in enumerate(synthetic_info):
                    if i >= j:
                        continue
                    
                    anchor_i = info_i.get('anchor')
                    anchor_j = info_j.get('anchor')
                    
                    if anchor_i is not None and anchor_j is not None:
                        # 如果锚点相连，合成节点也相连
                        if self.are_nodes_connected(original_graph, anchor_i, anchor_j):
                            syn_i = n_original + i
                            syn_j = n_original + j
                            edges.append([syn_i, syn_j])
                            edges.append([syn_j, syn_i])
                        
                        # 如果有共同邻居，也考虑连接
                        common_neighbors = self.find_common_neighbors(
                            original_graph, anchor_i, anchor_j
                        )
                        if len(common_neighbors) >= 2:  # 至少2个共同邻居
                            syn_i = n_original + i
                            syn_j = n_original + j
                            edges.append([syn_i, syn_j])
                            edges.append([syn_j, syn_i])
        
        return edges
    
    def determine_all_connections(self,
                                 original_graph,
                                 synthetic_features: torch.Tensor,
                                 synthetic_info: List[Dict]) -> torch.Tensor:
        """
        确定所有新边（包括合成节点之间和合成节点与原图之间）
        
        Returns:
            new_edges: [2, E] 的边索引张量
        """
        all_edges = []
        
        # 1. 合成节点与原图的连接
        edges_to_original = self.connect_to_original(
            original_graph, synthetic_features, synthetic_info
        )
        all_edges.extend(edges_to_original)
        
        # 2. 合成节点之间的连接
        edges_among_synthetic = self.connect_among_synthetic(
            synthetic_features, synthetic_info, original_graph
        )
        all_edges.extend(edges_among_synthetic)
        
        if all_edges:
            return torch.tensor(all_edges).T  # [2, E]
        else:
            return torch.zeros((2, 0), dtype=torch.long)

# ==================== 8. GAN训练器 ====================

class GANTrainer:
    """
    GAN训练器
    
    功能：训练生成器和判别器
    
    Args:
        node_feature_dim: 节点特征维度
        n_classes: 类别数
        device: 设备
        lr: 学习率
        beta1: Adam优化器beta1参数
        beta2: Adam优化器beta2参数
    """
    def __init__(self,
                 node_feature_dim: int = 1280,
                 n_classes: int = 10,
                 device: str = 'cuda',
                 lr: float = 1e-4,
                 beta1: float = 0.5,
                 beta2: float = 0.999):
        
        self.device = device
        self.node_feature_dim = node_feature_dim
        self.n_classes = n_classes
        
        # 初始化模型
        self.generator = TransformerProteinGenerator(
            node_feature_dim=node_feature_dim,
            n_classes=n_classes
        ).to(device)
        
        self.discriminator = TransformerProteinDiscriminator(
            node_feature_dim=node_feature_dim
        ).to(device)
        
        self.context_extractor = GraphContextExtractor(
            node_feature_dim=node_feature_dim
        ).to(device)
        
        # 优化器
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(beta1, beta2)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(beta1, beta2)
        )
        self.c_optimizer = torch.optim.Adam(
            self.context_extractor.parameters(), lr=lr, betas=(beta1, beta2)
        )
        
        # 损失函数
        self.criterion = nn.BCELoss()
    
    def train_step(self, real_graph):
        """
        单步训练
        
        Args:
            real_graph: 单个蛋白质图
        
        Returns:
            d_loss, g_loss: 判别器和生成器损失
        """
        real_graph = real_graph.to(self.device)
        
        # 准备数据
        real_nodes = real_graph.x.unsqueeze(0)  # [1, num_nodes, node_dim]
        batch_size = 1
        num_nodes = real_nodes.size(1)
        
        # 准备标签（如果没有真实标签，使用零向量）
        if hasattr(real_graph, 'node_labels'):
            # 使用第一个节点的标签作为图标签（简化）
            class_label = real_graph.node_labels[0].item()
            real_labels = F.one_hot(
                torch.tensor([class_label]), 
                num_classes=self.n_classes
            ).float().to(self.device)
        else:
            real_labels = torch.zeros(1, self.n_classes).to(self.device)
        
        # 提取图上下文
        context = self.context_extractor(real_graph)  # [1, context_dim]
        
        # ===== 训练判别器 =====
        self.d_optimizer.zero_grad()
        
        # 真实数据的损失
        real_output = self.discriminator(real_nodes)
        d_real_loss = self.criterion(real_output, torch.ones_like(real_output))
        
        # 生成假数据
        noise = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
        fake_nodes = self.generator(
            noise, real_labels, context, n_generate=num_nodes
        )
        
        fake_output = self.discriminator(fake_nodes.detach())
        d_fake_loss = self.criterion(fake_output, torch.zeros_like(fake_output))
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.d_optimizer.step()
        
        # ===== 训练生成器 =====
        self.g_optimizer.zero_grad()
        
        noise = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
        fake_nodes = self.generator(noise, real_labels, context, n_generate=num_nodes)
        fake_output = self.discriminator(fake_nodes)
        g_loss = self.criterion(fake_output, torch.ones_like(fake_output))
        
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.g_optimizer.step()
        
        return d_loss.item(), g_loss.item()
    
    def train(self, dataset: ProteinGraphDataset, epochs: int = 50, save_path: str = 'gan_model.pt'):
        """
        完整训练循环
        
        Args:
            dataset: 蛋白质图数据集
            epochs: 训练轮数
            save_path: 模型保存路径
        """
        print(f"开始训练，共 {len(dataset)} 个蛋白质图，{epochs} 轮")
        
        for epoch in range(epochs):
            total_d_loss = 0
            total_g_loss = 0
            
            # 打乱数据顺序
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            
            pbar = tqdm(indices, desc=f"Epoch {epoch+1}/{epochs}")
            for idx in pbar:
                graph = dataset[idx]
                d_loss, g_loss = self.train_step(graph)
                total_d_loss += d_loss
                total_g_loss += g_loss
                
                pbar.set_postfix({
                    'D_loss': f'{d_loss:.4f}',
                    'G_loss': f'{g_loss:.4f}'
                })
            
            avg_d_loss = total_d_loss / len(dataset)
            avg_g_loss = total_g_loss / len(dataset)
            
            print(f"Epoch {epoch+1}/{epochs} - 平均 D_loss: {avg_d_loss:.4f}, 平均 G_loss: {avg_g_loss:.4f}")
            
            # 每10轮保存一次模型
            if (epoch + 1) % 10 == 0:
                self.save_model(f"{save_path}_epoch{epoch+1}.pt")
        
        # 保存最终模型
        self.save_model(save_path)
        print(f"训练完成，模型已保存到 {save_path}")
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'context_extractor': self.context_extractor.state_dict()
        }, path)
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.context_extractor.load_state_dict(checkpoint['context_extractor'])
        print(f"模型已从 {path} 加载")

# ==================== 9. 图增强器 ====================

class ProteinGraphAugmentor:
    """
    蛋白质图增强器
    
    功能：使用训练好的GAN为每个蛋白质图生成合成节点
    
    Args:
        gan_trainer: 训练好的GANTrainer实例
        target_ratio: 目标增强比例（如1.3表示增加30%节点）
        connection_strategy: 连接策略
    """
    def __init__(self, 
                 gan_trainer: GANTrainer,
                 target_ratio: float = 1.3,
                 connection_strategy: str = 'data_driven'):
        self.gan = gan_trainer
        self.target_ratio = target_ratio
        self.connector = SyntheticNodeConnector(strategy=connection_strategy)
        self.device = gan_trainer.device
        
        # 设置模型为评估模式
        self.gan.generator.eval()
        self.gan.context_extractor.eval()
    
    def generate_nodes_for_protein(self, 
                                   protein_graph, 
                                   n_nodes: int,
                                   target_class: Optional[int] = None) -> Tuple[torch.Tensor, List[Dict]]:
        """
        为单个蛋白质图生成指定数量的新节点
        
        Args:
            protein_graph: 蛋白质图
            n_nodes: 需要生成的节点数
            target_class: 目标类别（None表示随机）
        
        Returns:
            synthetic_features: [n_nodes, feat_dim] 合成节点特征
            generation_info: 每个节点的生成信息（包含锚点等）
        """
        # 提取图上下文
        context = self.gan.context_extractor(protein_graph.to(self.device))  # [1, context_dim]
        
        # 准备类别条件
        if target_class is None:
            # 随机选择类别
            if hasattr(protein_graph, 'node_labels'):
                # 从原图标签分布中采样
                labels = protein_graph.node_labels
                class_probs = torch.bincount(labels).float() / len(labels)
                target_class = torch.multinomial(class_probs, 1).item()
            else:
                target_class = 0
        
        class_cond = F.one_hot(
            torch.tensor([target_class]), 
            num_classes=self.gan.n_classes
        ).float().to(self.device)
        
        # 分批生成（避免显存溢出）
        batch_size = 32
        synthetic_features = []
        generation_info = []
        
        with torch.no_grad():
            for i in range(0, n_nodes, batch_size):
                current_batch = min(batch_size, n_nodes - i)
                
                # 生成噪声
                noise = torch.randn(current_batch, self.gan.generator.latent_dim).to(self.device)
                class_batch = class_cond.repeat(current_batch, 1)
                context_batch = context.repeat(current_batch, 1)
                
                # 生成节点
                nodes = self.gan.generator(
                    noise, 
                    class_batch, 
                    context_batch,
                    n_generate=1
                )  # [current_batch, 1, feat_dim]
                
                synthetic_features.append(nodes.squeeze(1))
                
                # 记录生成信息（为每个节点随机选择锚点）
                for _ in range(current_batch):
                    anchor_idx = torch.randint(0, protein_graph.num_nodes, (1,)).item()
                    generation_info.append({
                        'anchor': anchor_idx,
                        'class': target_class
                    })
        
        return torch.cat(synthetic_features, dim=0), generation_info
    
    def add_nodes_to_graph(self, 
                          original_graph, 
                          synthetic_features: torch.Tensor,
                          generation_info: List[Dict]) -> object:
        """
        将合成节点添加到原图
        
        Args:
            original_graph: 原图
            synthetic_features: 合成节点特征 [n_new, feat_dim]
            generation_info: 生成信息
        
        Returns:
            augmented_graph: 增强后的图
        """
        # 创建新图的副本
        if hasattr(original_graph, 'clone'):
            new_graph = original_graph.clone()
        else:
            # 手动复制属性
            new_graph = type(original_graph)()
            for key, value in original_graph.__dict__.items():
                if key != 'x' and key != 'edge_index' and key != 'num_nodes':
                    setattr(new_graph, key, value)
        
        n_old = original_graph.num_nodes
        n_new = len(synthetic_features)
        
        # 1. 添加新节点特征
        if hasattr(original_graph, 'x'):
            new_graph.x = torch.cat([
                original_graph.x.cpu(),
                synthetic_features.cpu()
            ], dim=0)
        else:
            new_graph.x = synthetic_features.cpu()
        
        # 2. 确定新边
        new_edges = self.connector.determine_all_connections(
            original_graph,
            synthetic_features,
            generation_info
        )  # [2, E]
        
        # 3. 合并边
        if hasattr(original_graph, 'edge_index') and original_graph.edge_index is not None:
            new_graph.edge_index = torch.cat([
                original_graph.edge_index.cpu(),
                new_edges
            ], dim=1)
        else:
            new_graph.edge_index = new_edges
        
        # 4. 更新节点标签（如果有）
        if hasattr(original_graph, 'node_labels'):
            # 为新节点分配标签
            new_labels = torch.tensor([
                info['class'] for info in generation_info
            ], dtype=torch.long)
            new_graph.node_labels = torch.cat([
                original_graph.node_labels.cpu(),
                new_labels
            ])
        
        # 5. 更新节点数
        new_graph.num_nodes = n_old + n_new
        
        return new_graph
    
    def augment_all_graphs(self, 
                          dataset: ProteinGraphDataset,
                          output_path: str = '495_enhance_graph.pt') -> List:
        """
        增强所有蛋白质图
        
        Args:
            dataset: 原始数据集
            output_path: 输出文件路径
        
        Returns:
            augmented_graphs: 增强后的图列表
        """
        print(f"开始增强 {len(dataset)} 个蛋白质图，目标比例: {self.target_ratio}")
        
        augmented_graphs = []
        
        for idx, graph in enumerate(tqdm(dataset, desc="增强图中")):
            # 1. 计算需要生成的节点数
            original_count = graph.num_nodes
            target_count = int(original_count * self.target_ratio)
            nodes_to_generate = target_count - original_count
            
            if nodes_to_generate <= 0:
                # 不需要增强
                augmented_graphs.append(graph)
                print(f"图 {idx}: {original_count} 节点，无需增强")
                continue
            
            # 2. 生成新节点
            synthetic_features, generation_info = self.generate_nodes_for_protein(
                graph, 
                nodes_to_generate
            )
            
            # 3. 添加到原图
            augmented_graph = self.add_nodes_to_graph(
                graph,
                synthetic_features,
                generation_info
            )
            
            augmented_graphs.append(augmented_graph)
            
            print(f"图 {idx}: {original_count} → {augmented_graph.num_nodes} 节点 "
                  f"(增加了 {nodes_to_generate} 个)")
        
        # 4. 保存增强后的图
        torch.save(augmented_graphs, output_path)
        print(f"增强完成！所有 {len(augmented_graphs)} 个图已保存到 {output_path}")
        
        # 5. 统计信息
        self.print_statistics(dataset, augmented_graphs)
        
        return augmented_graphs
    
    def print_statistics(self, original_graphs, augmented_graphs):
        """打印增强统计信息"""
        original_sizes = [g.num_nodes for g in original_graphs]
        augmented_sizes = [g.num_nodes for g in augmented_graphs]
        
        print("\n========== 增强统计 ==========")
        print(f"原始图 - 最小: {min(original_sizes)}, 最大: {max(original_sizes)}, "
              f"平均: {sum(original_sizes)/len(original_sizes):.2f}")
        print(f"增强后 - 最小: {min(augmented_sizes)}, 最大: {max(augmented_sizes)}, "
              f"平均: {sum(augmented_sizes)/len(augmented_sizes):.2f}")
