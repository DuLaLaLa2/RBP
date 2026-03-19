"""
蛋白质图GAN模型定义文件
包含：Transformer生成器、判别器、上下文提取器、连接器等核心组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import List, Dict

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


# ==================== 2. Transformer生成器 ====================

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
                 node_feature_dim: int = 1408,
                 latent_dim: int = 100,
                 n_classes: int = 2,
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


# ==================== 3. Transformer判别器 ====================

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


# ==================== 4. 图上下文提取器 ====================

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
    def __init__(self, node_feature_dim: int = 1408, context_dim: int = 128):
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
    
    Input:
        original_graph: 原图
        synthetic_features: [n_syn, feat_dim] 合成节点特征
        synthetic_info: 每个合成节点的生成信息（包含锚点等）
    
    Output:
        edges: [2, E] 的边索引张量
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


# ==================== 6. 工具函数 ====================

def set_seed(seed=42):
    """设置随机种子以保证可重复性"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)