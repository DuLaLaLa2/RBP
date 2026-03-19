from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    node_feature_dim: int = 1408
    struct_dim: int = 128
    seq_dim: int = 1280
    n_classes: int = 2

    latent_dim: int = 128
    context_dim: int = 512

    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1

    disc_d_model: int = 256
    disc_nhead: int = 4
    disc_num_layers: int = 3


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    save_interval: int = 10

    device: str = 'cuda'
    output_dir: str = 'test/model'

    minority_sample_weight: float = 2.0


@dataclass
class AugmentationConfig:
    target_ratio: float = 0.4
    connection_strategy: str = 'knn_similarity'
    similarity_threshold: float = 0.5
    k_neighbors: int = 5
    generation_batch_size: int = 32
    augment_only_minority: bool = True