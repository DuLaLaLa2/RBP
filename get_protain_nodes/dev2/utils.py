"""
工具函数文件
包含：数据集类、统计函数、可视化辅助等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from collections import Counter
import os
import json

from config import ModelConfig, TrainingConfig, AugmentationConfig


# ==================== 1. 数据集类 ====================

class ProteinGraphDataset(Dataset):
    """
    蛋白质图数据集 - 适配你的数据格式
    
    Args:
        graph_path: .pt文件路径，包含所有蛋白质图
        config: ModelConfig对象
        minority_classes: 少数类列表，用于训练时重点关注
    """
    def __init__(self, 
                 graph_path: str, 
                 config: ModelConfig,
                 minority_classes: Optional[List[int]] = None):
        
        self.graphs = torch.load(graph_path, weights_only=False)
        self.config = config
        self.minority_classes = minority_classes or []
        
        print(f"加载了 {len(self.graphs)} 个蛋白质图")
        
        # 验证数据格式
        self._validate_graphs()
        
        # 计算类别统计信息
        self.class_stats = self._compute_class_statistics()
    
    def _validate_graphs(self):
        """验证每个图的格式"""
        for i, graph in enumerate(self.graphs):
            # 检查必要属性
            required_attrs = ['x', 'edge_index', 'y', 'num_nodes']
            for attr in required_attrs:
                if not hasattr(graph, attr):
                    raise ValueError(f"图 {i} 缺少必要属性: {attr}")
            
            # 检查维度
            if graph.x.size(1) != self.config.node_feature_dim:
                print(f"警告: 图 {i} 的特征维度是 {graph.x.size(1)}，"
                      f"期望是 {self.config.node_feature_dim}")
            
            # 确保y是long类型
            graph.y = graph.y.long()
            
            # 检查类别数
            if graph.y.max().item() >= self.config.n_classes:
                print(f"警告: 图 {i} 的类别 {graph.y.max().item()} "
                      f"超过了设定的类别数 {self.config.n_classes}")
    
    def _compute_class_statistics(self) -> Dict:
        """计算所有图的类别统计信息"""
        all_labels = []
        for graph in self.graphs:
            all_labels.extend(graph.y.tolist())
        
        counter = Counter(all_labels)
        
        stats = {
            'total_nodes': len(all_labels),
            'class_counts': dict(counter),
            'class_ratios': {k: v/len(all_labels) for k, v in counter.items()},
            'minority_classes': self.minority_classes
        }
        
        print("\n数据集统计:")
        print(f"总节点数: {stats['total_nodes']}")
        print("类别分布:")
        for class_id, count in sorted(stats['class_counts'].items()):
            ratio = stats['class_ratios'][class_id]
            mark = " (少数类)" if class_id in self.minority_classes else ""
            print(f"  类别 {class_id}: {count} 节点 ({ratio:.2%}){mark}")
        
        return stats
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int):
        return self.graphs[idx]


# ==================== 2. 数据增强辅助函数 ====================

class AugmentationHelper:
    """
    增强辅助类 - 计算需要生成的节点数等
    """
    
    @staticmethod
    def calculate_needed_nodes(graph, 
                               target_ratios: Dict[int, float]) -> Dict[int, int]:
        """
        计算每个类别需要生成的节点数
        
        Args:
            graph: 蛋白质图
            target_ratios: 目标比例，如 {0: 1.0, 1: 1.0} 表示1:1
        
        Returns:
            needed: 每个类别需要生成的节点数
        """
        # 统计当前各类别数量
        class_counts = torch.bincount(graph.y, minlength=max(target_ratios.keys())+1)
        
        # 以多数类为基准
        majority_count = class_counts.max().item()
        
        needed = {}
        for class_id, target_ratio in target_ratios.items():
            current = class_counts[class_id].item()
            target = int(majority_count * target_ratio)
            
            if current < target:
                needed[class_id] = target - current
        
        return needed
    
    @staticmethod
    def calculate_adaptive_ratio(graph, 
                                  base_ratio: float = 1.0,
                                  complexity_factor: float = 0.5) -> Dict[int, float]:
        """
        根据图复杂度自适应调整比例
        
        Args:
            graph: 蛋白质图
            base_ratio: 基础比例
            complexity_factor: 复杂度影响因子
        
        Returns:
            调整后的比例
        """
        # 计算图复杂度（基于节点数和边数）
        num_nodes = graph.num_nodes
        num_edges = graph.edge_index.size(1) if hasattr(graph, 'edge_index') else 0
        
        # 复杂度指标
        density = 2 * num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        complexity = min(1.0, density * 10)  # 归一化到 [0, 1]
        
        # 调整比例
        adjusted_ratio = base_ratio * (1 + complexity * complexity_factor)
        
        return adjusted_ratio
    
    @staticmethod
    def validate_augmentation(original_graph, augmented_graph) -> Dict:
        """
        验证增强效果
        
        Returns:
            包含各种指标的字典
        """
        metrics = {}
        
        # 1. 节点数变化
        metrics['original_nodes'] = original_graph.num_nodes
        metrics['augmented_nodes'] = augmented_graph.num_nodes
        metrics['node_increase'] = augmented_graph.num_nodes - original_graph.num_nodes
        metrics['node_increase_ratio'] = metrics['node_increase'] / original_graph.num_nodes
        
        # 2. 类别分布变化
        orig_counts = torch.bincount(original_graph.y)
        aug_counts = torch.bincount(augmented_graph.y)
        
        metrics['original_class_dist'] = orig_counts.tolist()
        metrics['augmented_class_dist'] = aug_counts.tolist()
        
        # 3. 边数变化
        if hasattr(original_graph, 'edge_index') and hasattr(augmented_graph, 'edge_index'):
            metrics['original_edges'] = original_graph.edge_index.size(1)
            metrics['augmented_edges'] = augmented_graph.edge_index.size(1)
            metrics['edge_increase'] = metrics['augmented_edges'] - metrics['original_edges']
        
        # 4. 特征分布相似性（仅比较原节点部分）
        orig_features = original_graph.x
        aug_orig_features = augmented_graph.x[:original_graph.num_nodes]
        
        # 计算均值差异
        metrics['feature_mean_diff'] = (orig_features.mean() - aug_orig_features.mean()).item()
        metrics['feature_std_diff'] = (orig_features.std() - aug_orig_features.std()).item()
        
        return metrics
    
    @staticmethod
    def print_augmentation_summary(metrics_list: List[Dict]):
        """打印增强结果汇总"""
        if not metrics_list:
            return
        
        total_original = sum(m['original_nodes'] for m in metrics_list)
        total_augmented = sum(m['augmented_nodes'] for m in metrics_list)
        total_increase = sum(m['node_increase'] for m in metrics_list)
        
        print("\n" + "="*50)
        print("增强结果汇总")
        print("="*50)
        print(f"处理图数量: {len(metrics_list)}")
        print(f"原始节点总数: {total_original}")
        print(f"增强后节点总数: {total_augmented}")
        print(f"新增节点总数: {total_increase}")
        print(f"平均增强比例: {(total_augmented/total_original - 1)*100:.2f}%")
        
        # 类别分布变化
        all_orig_counts = []
        all_aug_counts = []
        for m in metrics_list:
            all_orig_counts.extend(m['original_class_dist'])
            all_aug_counts.extend(m['augmented_class_dist'])
        
        print("\n类别分布变化:")
        # 这里需要更详细的类别统计


# ==================== 3. 模型保存/加载辅助 ====================

class ModelCheckpoint:
    """模型检查点管理"""
    
    def __init__(self, save_dir: str = './checkpoints'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save(self, 
             epoch: int,
             generator: nn.Module,
             discriminator: nn.Module,
             context_extractor: nn.Module,
             g_optimizer: torch.optim.Optimizer,
             d_optimizer: torch.optim.Optimizer,
             c_optimizer: torch.optim.Optimizer,
             config: dict,
             loss_history: dict,
             is_best: bool = False):
        """
        保存检查点
        """
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'context_extractor_state_dict': context_extractor.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'c_optimizer_state_dict': c_optimizer.state_dict(),
            'config': config,
            'loss_history': loss_history
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.save_dir, 'latest.pt')
        torch.save(checkpoint, latest_path)
        
        # 保存epoch检查点
        epoch_path = os.path.join(self.save_dir, f'epoch_{epoch}.pt')
        torch.save(checkpoint, epoch_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'best.pt')
            torch.save(checkpoint, best_path)
        
        print(f"检查点已保存: epoch {epoch}")
    
    def load(self, 
             path: str,
             generator: nn.Module,
             discriminator: nn.Module,
             context_extractor: nn.Module,
             g_optimizer: Optional[torch.optim.Optimizer] = None,
             d_optimizer: Optional[torch.optim.Optimizer] = None,
             c_optimizer: Optional[torch.optim.Optimizer] = None):
        """
        加载检查点
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"检查点文件不存在: {path}")
        
        checkpoint = torch.load(path)
        
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        context_extractor.load_state_dict(checkpoint['context_extractor_state_dict'])
        
        if g_optimizer is not None:
            g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        if d_optimizer is not None:
            d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        if c_optimizer is not None:
            c_optimizer.load_state_dict(checkpoint['c_optimizer_state_dict'])
        
        print(f"检查点已加载: {path}")
        print(f"训练轮数: {checkpoint['epoch']}")
        
        return checkpoint


# ==================== 4. 损失记录器 ====================

class LossHistory:
    """训练损失记录"""
    
    def __init__(self):
        self.history = {
            'd_loss': [],
            'g_loss': [],
            'epoch': []
        }
    
    def update(self, epoch: int, d_loss: float, g_loss: float):
        """更新损失"""
        self.history['epoch'].append(epoch)
        self.history['d_loss'].append(d_loss)
        self.history['g_loss'].append(g_loss)
    
    def get_latest(self) -> Dict:
        """获取最新损失"""
        if not self.history['epoch']:
            return {'d_loss': 0, 'g_loss': 0}
        
        return {
            'd_loss': self.history['d_loss'][-1],
            'g_loss': self.history['g_loss'][-1]
        }
    
    def get_average(self, window: int = 10) -> Dict:
        """获取平均损失"""
        if len(self.history['d_loss']) < window:
            return self.get_latest()
        
        return {
            'd_loss': sum(self.history['d_loss'][-window:]) / window,
            'g_loss': sum(self.history['g_loss'][-window:]) / window
        }
    
    def save(self, path: str):
        """保存历史"""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self, path: str):
        """加载历史"""
        with open(path, 'r') as f:
            self.history = json.load(f)


# ==================== 5. 随机种子设置 ====================

def set_seed(seed: int = 42):
    """
    设置所有随机种子以保证可重复性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"随机种子已设置为: {seed}")