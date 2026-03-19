"""
配置文件 - 所有超参数集中管理
"""
from dataclasses import dataclass
from typing import Dict, Optional, List

@dataclass
class ModelConfig:
    """模型配置 - 二分类版本"""
    # 数据维度
    node_feature_dim: int = 1408  # 128(struct) + 1280(seq)
    struct_dim: int = 128
    seq_dim: int = 1280
    n_classes: int = 2  # 固定为2类：0(非结合位点), 1(结合位点)
    
    # GAN配置
    latent_dim: int = 128
    context_dim: int = 256
    
    # Transformer配置
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    
    # 判别器配置
    disc_d_model: int = 256
    disc_nhead: int = 4
    disc_num_layers: int = 4


@dataclass
class TrainingConfig:
    """训练配置"""
    # 训练参数
    epochs: int = 100
    batch_size: int = 1  # 每个蛋白质图作为一个batch
    learning_rate: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    save_interval: int = 10
    
    # 设备配置
    device: str = 'cuda'  # 'cuda' 或 'cpu'
    
    # 输出配置
    output_dir: str = 'get_protain_nodes\\model'  # 模型保存目录
    
    # 结合位点（少数类）的采样权重，训练时多关注
    minority_sample_weight: float = 2.0


@dataclass
class AugmentationConfig:
    """增强配置 - 专门为结合位点设计"""
    # 目标比例：结合位点/非结合位点
    # 例如 1.0 表示完全平衡，0.5 表示结合位点达到非结合位点的一半
    target_ratio: float = 1.0
    
    # 连接策略
    connection_strategy: str = 'data_driven'  # 'independent', 'cluster', 'chain', 'fully_connected', 'data_driven'
    
    # 相似度阈值（用于cluster策略）
    similarity_threshold: float = 0.7
    
    # 生成批次大小
    generation_batch_size: int = 32
    
    # 是否只增强结合位点（少数类）
    augment_only_minority: bool = True