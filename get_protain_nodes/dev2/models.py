"""
模型定义文件
包含：Transformer生成器、判别器、上下文提取器、连接器等核心组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import List, Dict, Tuple, Optional, Union
import numpy as np

from config import ModelConfig


# ==================== 1. 位置编码模块 ====================

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
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
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
        """
        Args:
            x: [batch_size, seq_len, d_model]
        
        Returns:
            [batch_size, seq_len, d_model] 添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==================== 2. 图上下文提取器 ====================

class GraphContextExtractor(nn.Module):
    """
    从蛋白质图中提取全局上下文特征
    
    功能：将整个蛋白质图编码为一个固定长度的向量，用于条件生成
    
    Args:
        node_feature_dim: 节点特征维度
        context_dim: 输出上下文特征维度
    
    Input:
        protein_graph: 包含 .x (节点特征) 和 .edge_index 的图对象
    
    Output:
        context: [1, context_dim] 图上下文特征
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.node_feature_dim = config.node_feature_dim
        self.context_dim = config.context_dim
        
        # 图注意力层 - 用于聚合节点特征
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=config.node_feature_dim,
            num_heads=4,
            batch_first=True,
            dropout=config.dropout
        )
        
        # 全局特征提取器
        self.global_encoder = nn.Sequential(
            nn.Linear(config.node_feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # 统计特征编码器
        self.stat_encoder = nn.Sequential(
            nn.Linear(8, 64),  # 8个统计量
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        # 结构特征编码器（使用GCN捕捉图结构）
        self.structure_encoder = nn.Sequential(
            nn.Linear(config.node_feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128 + 128, 512),  # global + stats + structure
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.context_dim),
            nn.LayerNorm(config.context_dim)
        )
        
    def compute_graph_statistics(self, graph) -> torch.Tensor:
        """
        计算图的统计特征
        
        Returns:
            [1, 8] 统计特征
        """
        node_features = graph.x
        
        stats = []
        # 1. 特征均值
        stats.append(node_features.mean(dim=0).mean().item())
        # 2. 特征标准差
        stats.append(node_features.std(dim=0).mean().item())
        # 3. 特征最大值
        stats.append(node_features.max().item())
        # 4. 特征最小值
        stats.append(node_features.min().item())
        
        # 5. 平均度
        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
            num_edges = graph.edge_index.size(1)
            avg_degree = (2 * num_edges) / graph.num_nodes
        else:
            avg_degree = 0
        stats.append(avg_degree)
        
        # 6. 聚类系数
        clustering_coef = self.compute_clustering_coefficient(graph)
        stats.append(clustering_coef)
        
        # 7. 节点数
        stats.append(graph.num_nodes)
        
        # 8. 边数
        stats.append(graph.edge_index.size(1) if hasattr(graph, 'edge_index') else 0)
        
        return torch.tensor(stats).unsqueeze(0).to(node_features.device)
    
    def forward(self, protein_graph) -> torch.Tensor:
        """
        Args:
            protein_graph: 包含 .x 和 .edge_index 的图对象
        
        Returns:
            context: [1, context_dim] 图上下文特征
        """
        node_features = protein_graph.x  # [num_nodes, node_dim]
        device = node_features.device
        
        # 1. 全局特征（使用注意力池化）
        # 使用可学习的查询向量
        query = torch.randn(1, 1, self.node_feature_dim).to(device)
        attn_output, _ = self.graph_attention(
            query, 
            node_features.unsqueeze(0), 
            node_features.unsqueeze(0)
        )  # [1, 1, node_dim]
        
        global_feat = self.global_encoder(attn_output.squeeze(1))  # [1, 256]
        
        # 2. 统计特征
        stats_feat = self.compute_graph_statistics(protein_graph).to(device)  # [1, 8]
        stats_feat = self.stat_encoder(stats_feat)  # [1, 128]
        
        # 3. 结构特征（简化版 - 使用节点特征的均值代表结构）
        struct_feat = self.structure_encoder(node_features.mean(dim=0, keepdim=True))  # [1, 128]
        
        # 4. 融合所有特征
        combined = torch.cat([global_feat, stats_feat, struct_feat], dim=1)  # [1, 256+128+128=512]
        context = self.fusion(combined)  # [1, context_dim]
        
        return context
    
    def compute_clustering_coefficient(self, graph) -> float:
        """
        计算图的聚类系数
        
        聚类系数定义：对于每个节点，其邻居之间实际存在的边数与可能存在的边数的比值
        整个图的聚类系数是所有节点聚类系数的平均值
        
        Args:
            graph: 蛋白质图
        
        Returns:
            float: 图的聚类系数
        """
        if not hasattr(graph, 'edge_index') or graph.edge_index is None:
            return 0.0
        
        # 构建邻接表
        adjacency = {}
        edges = graph.edge_index
        
        # 填充邻接表
        for i in range(edges.size(1)):
            u = edges[0, i].item()
            v = edges[1, i].item()
            
            if u not in adjacency:
                adjacency[u] = set()
            if v not in adjacency:
                adjacency[v] = set()
            
            adjacency[u].add(v)
            adjacency[v].add(u)
        
        # 计算每个节点的聚类系数
        clustering_coefficients = []
        for node in adjacency:
            neighbors = adjacency[node]
            k = len(neighbors)
            
            if k < 2:
                # 度数小于2的节点聚类系数为0
                clustering_coefficients.append(0.0)
                continue
            
            # 计算邻居之间的边数
            edges_between_neighbors = 0
            neighbor_list = list(neighbors)
            
            for i in range(len(neighbor_list)):
                for j in range(i + 1, len(neighbor_list)):
                    if neighbor_list[j] in adjacency.get(neighbor_list[i], set()):
                        edges_between_neighbors += 1
            
            # 计算该节点的聚类系数
            max_possible_edges = k * (k - 1) / 2
            if max_possible_edges > 0:
                clustering_coefficients.append(edges_between_neighbors / max_possible_edges)
            else:
                clustering_coefficients.append(0.0)
        
        # 计算整个图的平均聚类系数
        if clustering_coefficients:
            return sum(clustering_coefficients) / len(clustering_coefficients)
        else:
            return 0.0


# ==================== 3. Transformer生成器 ====================

class TransformerProteinGenerator(nn.Module):
    """
    基于Transformer的蛋白质节点生成器
    
    功能：从随机噪声和条件信息生成新的蛋白质节点特征
    
    Args:
        config: ModelConfig对象
    
    Input:
        noise: [batch_size, latent_dim] 随机噪声
        class_cond: [batch_size, n_classes] 类别条件（one-hot）
        context: [batch_size, context_dim] 图上下文特征
        n_generate: 要生成的节点数量
    
    Output:
        generated_nodes: [batch_size, n_generate, node_feature_dim]
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.node_feature_dim = config.node_feature_dim
        self.latent_dim = config.latent_dim
        self.n_classes = config.n_classes
        self.context_dim = config.context_dim
        self.d_model = config.d_model
        
        # 输入投影层
        self.noise_proj = nn.Linear(config.latent_dim, config.d_model)
        self.class_proj = nn.Linear(config.n_classes, config.d_model)
        self.context_proj = nn.Linear(config.context_dim, config.d_model)
        
        # 可学习的节点提示（用于生成多个节点时的初始化）
        self.node_prompts = nn.Parameter(torch.randn(1, 100, config.d_model) * 0.02)
        
        # 位置编码
        self.position_encoding = PositionalEncoding(
            config.d_model, 
            config.dropout, 
            max_len=2000
        )
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm 结构，训练更稳定
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )
        
        # 输出投影（多层）
        self.output_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.LayerNorm(config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.d_model * 2, config.d_model * 2),
            nn.LayerNorm(config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.d_model * 2, config.node_feature_dim)
        )
        
        # 输出层归一化
        self.output_norm = nn.LayerNorm(config.node_feature_dim)
        
        # 自适应实例归一化参数预测 (增加中间层维度)
        self.adain_predictor = nn.Sequential(
            nn.Linear(config.context_dim, 512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, config.node_feature_dim * 2)  # gamma and beta
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重 - Xavier初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1 and p.size(0) == self.node_feature_dim:
                # 对偏置项用零初始化
                nn.init.zeros_(p)
    
    def adain(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        自适应实例归一化
        
        Args:
            x: [batch_size, n_generate, node_feature_dim] 或 [batch_size, node_feature_dim]
            context: [batch_size, context_dim]
        
        Returns:
            归一化后的张量
        """
        # 预测gamma和beta
        params = self.adain_predictor(context)  # [batch_size, node_feature_dim*2]
        gamma, beta = torch.chunk(params, 2, dim=1)  # 各 [batch_size, node_feature_dim]
        
        # 调整维度以匹配x
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)  # [batch_size, 1, node_feature_dim]
            beta = beta.unsqueeze(1)
        
        # 计算均值和标准差
        if x.dim() == 3:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True) + 1e-5
        else:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True) + 1e-5
        
        # 应用自适应归一化
        x_norm = (x - mean) / std
        x_norm = torch.clamp(x_norm, min=-10, max=10)
        return x_norm * gamma + beta
    
    def forward(self, 
                noise: torch.Tensor, 
                class_cond: torch.Tensor, 
                context: torch.Tensor,
                n_generate: int = 1) -> torch.Tensor:
        """
        Args:
            noise: [batch_size, latent_dim] 随机噪声
            class_cond: [batch_size, n_classes] 类别条件
            context: [batch_size, context_dim] 图上下文
            n_generate: 要生成的节点数量
        
        Returns:
            generated_nodes: [batch_size, n_generate, node_feature_dim]
        """
        batch_size = noise.size(0)
        
        # 1. 投影所有输入到d_model维度
        noise_feat = self.noise_proj(noise).unsqueeze(1)  # [batch_size, 1, d_model]
        class_feat = self.class_proj(class_cond).unsqueeze(1)  # [batch_size, 1, d_model]
        context_feat = self.context_proj(context).unsqueeze(1)  # [batch_size, 1, d_model]
        
        # 2. 融合条件特征
        cond_feat = noise_feat + class_feat + context_feat  # [batch_size, 1, d_model]
        
        # 3. 准备序列输入（使用节点提示）
        if n_generate > 1:
            # 复制节点提示
            prompts = self.node_prompts[:, :n_generate, :].expand(batch_size, -1, -1)
            # 将条件特征与提示结合
            seq_input = prompts + cond_feat
        else:
            seq_input = cond_feat
        
        # 4. 添加位置编码
        seq_input = self.position_encoding(seq_input)
        
        # 5. Transformer编码
        transformer_output = self.transformer_encoder(seq_input)  # [batch_size, n_generate, d_model]
        
        # 6. 输出投影
        raw_output = self.output_proj(transformer_output)  # [batch_size, n_generate, node_feature_dim]
        
        # 7. 自适应归一化
        output = self.adain(raw_output, context)
        
        # 8. 输出层归一化
        output = self.output_norm(output)
        
        return output


# ==================== 4. Transformer判别器 ====================

class TransformerProteinDiscriminator(nn.Module):
    """
    基于Transformer的蛋白质节点判别器
    
    功能：判断节点是真实的还是生成的
    
    Args:
        config: ModelConfig对象
    
    Input:
        node_features: [batch_size, num_nodes, node_feature_dim] 节点特征
    
    Output:
        [batch_size, 1] 真实性概率（0-1之间）
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.node_feature_dim = config.node_feature_dim
        self.d_model = config.disc_d_model
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(config.node_feature_dim, config.disc_d_model),
            nn.LayerNorm(config.disc_d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # 位置编码
        self.position_encoding = PositionalEncoding(
            config.disc_d_model, 
            config.dropout, 
            max_len=2000
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.disc_d_model,
            nhead=config.disc_nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.disc_num_layers
        )
        
        # 图级别池化（使用注意力）
        self.pooling_attention = nn.MultiheadAttention(
            embed_dim=config.disc_d_model,
            num_heads=config.disc_nhead,
            batch_first=True,
            dropout=config.dropout
        )
        
        # 可学习的池化查询
        self.pool_query = nn.Parameter(torch.randn(1, 1, config.disc_d_model) * 0.02)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.disc_d_model, config.disc_d_model // 2),
            nn.LayerNorm(config.disc_d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout * 2),
            
            nn.Linear(config.disc_d_model // 2, config.disc_d_model // 4),
            nn.LayerNorm(config.disc_d_model // 4),
            nn.GELU(),
            nn.Dropout(config.dropout * 2),
            
            nn.Linear(config.disc_d_model // 4, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [batch_size, num_nodes, node_feature_dim]
        
        Returns:
            [batch_size, 1] 真实性概率
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # 1. 输入投影
        x = self.input_proj(node_features)  # [batch_size, num_nodes, d_model]
        
        # 2. 添加位置编码
        x = self.position_encoding(x)
        
        # 3. Transformer编码
        x = self.transformer_encoder(x)  # [batch_size, num_nodes, d_model]
        
        # 4. 注意力池化
        query = self.pool_query.expand(batch_size, -1, -1)  # [batch_size, 1, d_model]
        pooled, attention_weights = self.pooling_attention(
            query, x, x
        )  # [batch_size, 1, d_model]
        
        # 5. 分类
        out = self.classifier(pooled.squeeze(1))  # [batch_size, 1]
        
        return out
    
    def get_intermediate_features(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        获取判别器的中间特征用于特征匹配损失
        
        Args:
            node_features: [batch_size, num_nodes, node_feature_dim]
        
        Returns:
            [batch_size, d_model] 中间特征
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # 1. 输入投影
        x = self.input_proj(node_features)  # [batch_size, num_nodes, d_model]
        
        # 2. 添加位置编码
        x = self.position_encoding(x)
        
        # 3. Transformer编码
        x = self.transformer_encoder(x)  # [batch_size, num_nodes, d_model]
        
        # 4. 注意力池化
        query = self.pool_query.expand(batch_size, -1, -1)  # [batch_size, 1, d_model]
        pooled, _ = self.pooling_attention(
            query, x, x
        )  # [batch_size, 1, d_model]
        
        return pooled.squeeze(1)  # [batch_size, d_model]


# ==================== 5. 合成节点连接器 ====================

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
        connection_similarity_threshold: 连接相似度阈值（用于与原图连接）
        max_connections: 最大连接数
    """
    def __init__(self, 
                 strategy: str = 'data_driven', 
                 similarity_threshold: float = 0.7,
                 connection_similarity_threshold: float = 0.6,
                 max_connections: int = 4):
        self.strategy = strategy
        self.similarity_threshold = similarity_threshold
        self.connection_similarity_threshold = connection_similarity_threshold
        self.max_connections = max_connections
        
        # 策略验证
        valid_strategies = ['independent', 'cluster', 'chain', 'fully_connected', 'data_driven']
        if strategy not in valid_strategies:
            raise ValueError(f"策略必须为以下之一: {valid_strategies}")
    
    def compute_similarity(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """计算两个节点特征的余弦相似度"""
        return F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()
    
    def compute_euclidean_distance(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """计算欧氏距离"""
        return torch.norm(feat1 - feat2).item()
    
    def are_nodes_connected(self, graph, node1: int, node2: int) -> bool:
        """检查原图中两个节点是否相连"""
        if not hasattr(graph, 'edge_index') or graph.edge_index is None:
            return False
        
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
    
    def get_node_degree(self, graph, node: int) -> int:
        """获取节点的度"""
        if not hasattr(graph, 'edge_index') or graph.edge_index is None:
            return 0
        
        edges = graph.edge_index
        degree = 0
        for i in range(edges.size(1)):
            if edges[0, i] == node or edges[1, i] == node:
                degree += 1
        return degree
    
    def connect_to_original(self, 
                           original_graph, 
                           synthetic_features: torch.Tensor,
                           synthetic_info: List[Dict]) -> List[List[int]]:
        """
        确定合成节点与原图节点的连接
        
        基于特征相似度和结构相似度
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
            
            # 根据原图节点的度决定连接数量
            # 高度节点连接更多新节点
            for j, (orig_node, sim) in enumerate(similarities[:5]):  # 最多考虑前5个
                if sim > self.connection_similarity_threshold:  # 相似度阈值
                    orig_degree = self.get_node_degree(original_graph, orig_node)
                    
                    # 连接概率与相似度和原节点度相关
                    connect_prob = sim * min(1.0, orig_degree / 10.0)
                    
                    if random.random() < connect_prob:
                        syn_node = n_original + i
                        edges.append([syn_node, orig_node])
                        edges.append([orig_node, syn_node])
                        
                        # 如果已经连接了足够多的节点，就停止
                        if len([e for e in edges if e[0] == syn_node or e[1] == syn_node]) >= self.max_connections:
                            break
        
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
            # 策略5：基于锚点关系和特征相似度
            for i, info_i in enumerate(synthetic_info):
                for j, info_j in enumerate(synthetic_info):
                    if i >= j:
                        continue
                    
                    anchor_i = info_i.get('anchor')
                    anchor_j = info_j.get('anchor')
                    
                    # 计算合成节点特征相似度
                    feature_similarity = self.compute_similarity(
                        synthetic_features[i], 
                        synthetic_features[j]
                    )
                    
                    # 基于多种因素决定是否连接
                    should_connect = False
                    
                    if anchor_i is not None and anchor_j is not None:
                        # 因素1：锚点相连
                        if self.are_nodes_connected(original_graph, anchor_i, anchor_j):
                            should_connect = True
                        
                        # 因素2：有共同邻居
                        common_neighbors = self.find_common_neighbors(
                            original_graph, anchor_i, anchor_j
                        )
                        if len(common_neighbors) >= 2:
                            should_connect = True
                        
                        # 因素3：锚点的度相似
                        degree_i = self.get_node_degree(original_graph, anchor_i)
                        degree_j = self.get_node_degree(original_graph, anchor_j)
                        if abs(degree_i - degree_j) <= 2 and degree_i > 0:
                            should_connect = True
                    
                    # 因素4：特征相似度高
                    if feature_similarity > self.similarity_threshold:
                        should_connect = True
                    
                    # 连接节点
                    if should_connect:
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
        
        # 去重
        if all_edges:
            # 转换为集合去重
            edge_set = set()
            for u, v in all_edges:
                if u != v:  # 避免自环
                    # 存储为元组，确保无向边只存一次
                    edge_set.add(tuple(sorted([u, v])))
            
            # 转换回列表，并生成双向边
            unique_edges = []
            for u, v in edge_set:
                unique_edges.append([u, v])
                unique_edges.append([v, u])
            
            return torch.tensor(unique_edges).T  # [2, E]
        else:
            return torch.zeros((2, 0), dtype=torch.long)