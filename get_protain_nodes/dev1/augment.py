"""
增强脚本 - 使用训练好的模型为新的蛋白质图生成增强节点
可以批量处理任意数量的新蛋白质图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Union
from tqdm import tqdm
import argparse
import os

from protein_gan_models import (
    TransformerProteinGenerator,
    TransformerProteinDiscriminator,
    GraphContextExtractor,
    SyntheticNodeConnector,
    set_seed
)


class ProteinGraphAugmentor:
    """
    蛋白质图增强器 - 使用训练好的模型为新图生成节点
    
    功能：
    1. 加载训练好的GAN模型
    2. 为单个或多个新蛋白质图生成增强节点
    3. 支持批量处理和自定义增强比例
    """
    def __init__(self, 
                 model_path: str,
                 device: str = 'cuda'):
        """
        Args:
            model_path: 训练好的模型文件路径
            device: 设备
        """
        self.device = device
        self.model_path = model_path
        
        # 加载模型和配置
        self._load_model()
        
        # 设置模型为评估模式
        self.generator.eval()
        self.context_extractor.eval()
        
        print(f"增强器初始化完成，使用设备: {device}")
        print(f"模型配置: 特征维度={self.node_feature_dim}, 类别数={self.n_classes}")
    
    def _load_model(self):
        """加载训练好的模型"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 获取模型配置
        config = checkpoint.get('model_config', {})
        self.node_feature_dim = config.get('node_feature_dim', 1280)
        self.n_classes = config.get('n_classes', 10)
        self.latent_dim = config.get('latent_dim', 100)
        
        # 初始化模型
        self.generator = TransformerProteinGenerator(
            node_feature_dim=self.node_feature_dim,
            latent_dim=self.latent_dim,
            n_classes=self.n_classes
        ).to(self.device)
        
        self.discriminator = TransformerProteinDiscriminator(
            node_feature_dim=self.node_feature_dim
        ).to(self.device)
        
        self.context_extractor = GraphContextExtractor(
            node_feature_dim=self.node_feature_dim
        ).to(self.device)
        
        # 加载权重
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.context_extractor.load_state_dict(checkpoint['context_extractor'])
        
        print(f"模型加载成功: {self.model_path}")
    
    def generate_nodes_for_protein(self, 
                                   protein_graph,
                                   n_nodes: int,
                                   target_class: Optional[int] = None,
                                   batch_size: int = 32) -> tuple:
        """
        为单个蛋白质图生成指定数量的新节点
        
        Args:
            protein_graph: 单个蛋白质图对象
            n_nodes: 需要生成的节点数
            target_class: 目标类别（None表示从原图分布采样）
            batch_size: 生成批次大小
        
        Returns:
            synthetic_features: [n_nodes, feat_dim] 合成节点特征
            generation_info: 每个节点的生成信息
        """
        # 提取图上下文
        context = self.context_extractor(protein_graph.to(self.device))  # [1, context_dim]
        
        # 准备类别条件
        if target_class is None:
            # 从原图标签分布中采样
            if hasattr(protein_graph, 'y'):
                labels = protein_graph.y
                class_probs = torch.bincount(labels).float() / len(labels)
                target_class = torch.multinomial(class_probs, 1).item()
            else:
                target_class = 0
        
        class_cond = F.one_hot(
            torch.tensor([target_class]), 
            num_classes=self.n_classes
        ).float().to(self.device)
        
        # 分批生成
        synthetic_features = []
        generation_info = []
        
        with torch.no_grad():
            for i in range(0, n_nodes, batch_size):
                current_batch = min(batch_size, n_nodes - i)
                
                noise = torch.randn(current_batch, self.latent_dim).to(self.device)
                class_batch = class_cond.repeat(current_batch, 1)
                context_batch = context.repeat(current_batch, 1)
                
                nodes = self.generator(
                    noise, 
                    class_batch, 
                    context_batch,
                    n_generate=1
                )  # [current_batch, 1, feat_dim]
                
                synthetic_features.append(nodes.squeeze(1).cpu())
                
                # 记录生成信息
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
                          generation_info: List[Dict],
                          connection_strategy: str = 'data_driven') -> object:
        """
        将合成节点添加到原图
        
        Args:
            original_graph: 原图
            synthetic_features: 合成节点特征
            generation_info: 生成信息
            connection_strategy: 连接策略
        
        Returns:
            augmented_graph: 增强后的图
        """
        connector = SyntheticNodeConnector(strategy=connection_strategy)
        
        # 创建新图的副本
        if hasattr(original_graph, 'clone'):
            new_graph = original_graph.clone()
        else:
            new_graph = type(original_graph)()
            for key, value in original_graph.__dict__.items():
                if key not in ['x', 'edge_index', 'num_nodes', 'y', 'struct_feat', 'seq_feat', 'pos']:
                    setattr(new_graph, key, value)
        
        n_old = original_graph.num_nodes
        n_new = len(synthetic_features)
        
        # 添加新节点特征
        if hasattr(original_graph, 'x'):
            new_graph.x = torch.cat([
                original_graph.x.cpu(),
                synthetic_features.cpu()
            ], dim=0)
        else:
            new_graph.x = synthetic_features.cpu()
        
        # 确定新边
        new_edges = connector.determine_all_connections(
            original_graph,
            synthetic_features,
            generation_info
        )
        
        # 合并边
        if hasattr(original_graph, 'edge_index') and original_graph.edge_index is not None:
            new_graph.edge_index = torch.cat([
                original_graph.edge_index.cpu(),
                new_edges
            ], dim=1)
        else:
            new_graph.edge_index = new_edges
        
        # 更新节点标签
        if hasattr(original_graph, 'y'):
            new_labels = torch.tensor([
                info['class'] for info in generation_info
            ], dtype=torch.long)
            new_graph.node_labels = torch.cat([
                original_graph.node_labels.cpu(),
                new_labels
            ])
        
        # 更新节点数
        new_graph.num_nodes = n_old + n_new
        
        return new_graph
    
    def augment_single_graph(self,
                            protein_graph,
                            target_ratio: float = 1.3,
                            target_class: Optional[int] = None,
                            connection_strategy: str = 'data_driven') -> object:
        """
        增强单个蛋白质图
        
        Args:
            protein_graph: 单个蛋白质图
            target_ratio: 目标增强比例
            target_class: 目标类别
            connection_strategy: 连接策略
        
        Returns:
            augmented_graph: 增强后的图
        """
        original_count = protein_graph.num_nodes
        target_count = int(original_count * target_ratio)
        nodes_to_generate = target_count - original_count
        
        if nodes_to_generate <= 0:
            print(f"无需增强，原图已有 {original_count} 节点")
            return protein_graph
        
        # 生成新节点
        synthetic_features, generation_info = self.generate_nodes_for_protein(
            protein_graph,
            nodes_to_generate,
            target_class
        )
        
        # 添加到原图
        augmented_graph = self.add_nodes_to_graph(
            protein_graph,
            synthetic_features,
            generation_info,
            connection_strategy
        )
        
        print(f"增强完成: {original_count} → {augmented_graph.num_nodes} 节点 "
              f"(增加了 {nodes_to_generate} 个)")
        
        return augmented_graph
    
    def augment_multiple_graphs(self,
                               graphs: List,
                               target_ratio: float = 1.3,
                               target_class: Optional[int] = None,
                               connection_strategy: str = 'data_driven',
                               output_path: Optional[str] = None) -> List:
        """
        批量增强多个蛋白质图
        
        Args:
            graphs: 蛋白质图列表
            target_ratio: 目标增强比例
            target_class: 目标类别
            connection_strategy: 连接策略
            output_path: 输出文件路径（可选）
        
        Returns:
            augmented_graphs: 增强后的图列表
        """
        print(f"开始批量增强 {len(graphs)} 个蛋白质图，目标比例: {target_ratio}")
        
        augmented_graphs = []
        stats = {
            'original_nodes': [],
            'augmented_nodes': [],
            'generated_nodes': []
        }
        
        for idx, graph in enumerate(tqdm(graphs, desc="增强图中")):
            original_count = graph.num_nodes
            target_count = int(original_count * target_ratio)
            nodes_to_generate = target_count - original_count
            
            if nodes_to_generate <= 0:
                augmented_graphs.append(graph)
                stats['original_nodes'].append(original_count)
                stats['augmented_nodes'].append(original_count)
                stats['generated_nodes'].append(0)
                continue
            
            # 生成新节点
            synthetic_features, generation_info = self.generate_nodes_for_protein(
                graph,
                nodes_to_generate,
                target_class
            )
            
            # 添加到原图
            augmented_graph = self.add_nodes_to_graph(
                graph,
                synthetic_features,
                generation_info,
                connection_strategy
            )
            
            augmented_graphs.append(augmented_graph)
            
            stats['original_nodes'].append(original_count)
            stats['augmented_nodes'].append(augmented_graph.num_nodes)
            stats['generated_nodes'].append(nodes_to_generate)
        
        # 打印统计信息
        self._print_statistics(stats)
        
        # 保存结果
        if output_path:
            torch.save(augmented_graphs, output_path)
            print(f"增强后的图已保存到: {output_path}")
        
        return augmented_graphs
    
    def _print_statistics(self, stats: Dict):
        """打印增强统计信息"""
        original_total = sum(stats['original_nodes'])
        augmented_total = sum(stats['augmented_nodes'])
        generated_total = sum(stats['generated_nodes'])
        
        print("\n========== 增强统计 ==========")
        print(f"处理图数量: {len(stats['original_nodes'])}")
        print(f"原始节点总数: {original_total}")
        print(f"增强后节点总数: {augmented_total}")
        print(f"生成节点总数: {generated_total}")
        print(f"平均增强比例: {(augmented_total/original_total - 1)*100:.2f}%")
        
        if stats['original_nodes']:
            print(f"\n节点数范围:")
            print(f"  原始: {min(stats['original_nodes'])} - {max(stats['original_nodes'])}")
            print(f"  增强后: {min(stats['augmented_nodes'])} - {max(stats['augmented_nodes'])}")


def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型增强蛋白质图')
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型文件路径')
    parser.add_argument('--input_path', type=str, required=True,
                        help='输入蛋白质图文件路径 (.pt)')
    parser.add_argument('--output_path', type=str, default='enhanced_graphs.pt',
                        help='输出增强后文件路径')
    parser.add_argument('--target_ratio', type=float, default=1.3,
                        help='目标增强比例')
    parser.add_argument('--target_class', type=int, default=None,
                        help='目标类别（None表示从原图分布采样）')
    parser.add_argument('--connection_strategy', type=str, default='data_driven',
                        choices=['independent', 'cluster', 'chain', 'fully_connected', 'data_driven'],
                        help='连接策略')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"使用设备: {device}")
    
    # 1. 创建增强器
    print(f"\n=== 初始化增强器 ===")
    augmentor = ProteinGraphAugmentor(
        model_path=args.model_path,
        device=device
    )
    
    # 2. 加载新蛋白质图
    print(f"\n=== 加载新蛋白质图 ===")
    new_graphs = torch.load(args.input_path)
    print(f"加载了 {len(new_graphs)} 个新蛋白质图")
    
    # 3. 批量增强
    print(f"\n=== 开始增强 ===")
    augmented_graphs = augmentor.augment_multiple_graphs(
        graphs=new_graphs,
        target_ratio=args.target_ratio,
        target_class=args.target_class,
        connection_strategy=args.connection_strategy,
        output_path=args.output_path
    )
    
    print(f"\n=== 完成！ ===")


if __name__ == "__main__":
    main()