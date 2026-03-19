"""
增强脚本 - 使用训练好的模型为蛋白质图生成结合位点
直接在代码中配置参数，无需命令行
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
from typing import List, Dict, Optional

from config import ModelConfig, AugmentationConfig
from models import (
    TransformerProteinGenerator,
    TransformerProteinDiscriminator,
    GraphContextExtractor,
    SyntheticNodeConnector
)
from utils import (
    AugmentationHelper,
    set_seed
)


# ==================== 在这里直接配置增强参数 ====================

# 模型路径
MODEL_PATH = r"./checkpoints/best.pt"  # 训练好的模型路径

# 数据路径
INPUT_PATH = r"D:\\your_path\\new_protein_graphs.pt"  # 需要增强的蛋白质图
OUTPUT_PATH = r"./enhanced_graphs.pt"                # 增强后的输出路径

# 增强配置
TARGET_RATIO = 1.0           # 目标比例：结合位点/非结合位点，1.0表示完全平衡
CONNECTION_STRATEGY = 'data_driven'  # 连接策略
SIMILARITY_THRESHOLD = 0.7   # 相似度阈值

# 其他配置
SEED = 42                    # 随机种子
DEVICE = 'cuda'              # 设备

# ==============================================================


class ProteinGraphAugmentor:
    """
    蛋白质图增强器 - 专门为结合位点（类别1）生成节点
    """
    
    def __init__(self, 
                 model_path: str,
                 model_config: ModelConfig,
                 aug_config: AugmentationConfig,
                 device: str = 'cuda'):
        
        self.model_config = model_config
        self.aug_config = aug_config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self._load_model(model_path)
        
        # 创建连接器
        self.connector = SyntheticNodeConnector(
            strategy=aug_config.connection_strategy,
            similarity_threshold=aug_config.similarity_threshold
        )
        
        print(f"\n{'='*50}")
        print("增强器配置")
        print(f"{'='*50}")
        print(f"目标比例: 结合位点/非结合位点 = {aug_config.target_ratio}")
        print(f"连接策略: {aug_config.connection_strategy}")
        print(f"设备: {self.device}")
    
    def _load_model(self, model_path: str):
        """加载训练好的模型"""
        print(f"加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.generator = TransformerProteinGenerator(self.model_config).to(self.device)
        self.discriminator = TransformerProteinDiscriminator(self.model_config).to(self.device)
        self.context_extractor = GraphContextExtractor(self.model_config).to(self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.context_extractor.load_state_dict(checkpoint['context_extractor_state_dict'])
        
        self.generator.eval()
        self.discriminator.eval()
        self.context_extractor.eval()
        
        print("模型加载成功！")
    
    def generate_binding_sites(self,
                              protein_graph,
                              n_nodes: int) -> tuple:
        """
        生成结合位点（类别1）
        """
        # 提取图上下文
        context = self.context_extractor(protein_graph.to(self.device))
        
        # 准备类别条件（固定为类别1：结合位点）
        class_cond = F.one_hot(
            torch.tensor([1]),
            num_classes=2
        ).float().to(self.device)
        
        # 分批生成
        synthetic_features = []
        generation_info = []
        batch_size = self.aug_config.generation_batch_size
        
        with torch.no_grad():
            for i in range(0, n_nodes, batch_size):
                current_batch = min(batch_size, n_nodes - i)
                
                noise = torch.randn(current_batch, self.model_config.latent_dim).to(self.device)
                class_batch = class_cond.repeat(current_batch, 1)
                context_batch = context.repeat(current_batch, 1)
                
                nodes = self.generator(
                    noise, 
                    class_batch, 
                    context_batch,
                    n_generate=1
                )
                
                synthetic_features.append(nodes.squeeze(1).cpu())
                
                # 记录生成信息
                for j in range(current_batch):
                    # 随机选择锚点
                    if (protein_graph.y == 1).any():
                        binding_sites = torch.where(protein_graph.y == 1)[0]
                        anchor_idx = binding_sites[torch.randint(0, len(binding_sites), (1,))].item()
                    else:
                        anchor_idx = torch.randint(0, protein_graph.num_nodes, (1,)).item()
                    
                    generation_info.append({
                        'anchor': anchor_idx,
                        'class': 1,
                        'batch_idx': i + j
                    })
        
        return torch.cat(synthetic_features, dim=0), generation_info
    
    def calculate_needed_binding_sites(self, protein_graph) -> int:
        """
        计算需要生成的结合位点数量
        """
        class_counts = torch.bincount(protein_graph.y, minlength=2)
        non_binding = class_counts[0].item()
        binding = class_counts[1].item()
        
        target_binding = int(non_binding * self.aug_config.target_ratio)
        needed = max(0, target_binding - binding)
        
        print(f"\n当前统计:")
        print(f"  非结合位点: {non_binding}")
        print(f"  结合位点: {binding}")
        print(f"  当前比例: {binding/non_binding:.3f}")
        print(f"  目标比例: {self.aug_config.target_ratio}")
        print(f"  需要生成: {needed} 个结合位点")
        
        return needed
    
    def add_binding_sites_to_graph(self,
                                  original_graph,
                                  synthetic_features: torch.Tensor,
                                  generation_info: List[Dict]) -> object:
        """
        将生成的结合位点添加到原图
        """
        if hasattr(original_graph, 'clone'):
            new_graph = original_graph.clone()
        else:
            new_graph = type(original_graph)()
            for key, value in original_graph.__dict__.items():
                if not key.startswith('_'):
                    setattr(new_graph, key, value)
        
        n_old = original_graph.num_nodes
        n_new = len(synthetic_features)
        
        # 添加新节点特征
        new_graph.x = torch.cat([
            original_graph.x.cpu(),
            synthetic_features.cpu()
        ], dim=0)
        
        # 分离结构特征和序列特征
        if hasattr(original_graph, 'struct_feat') and hasattr(original_graph, 'seq_feat'):
            new_struct = synthetic_features[:, :self.model_config.struct_dim].cpu()
            new_seq = synthetic_features[:, self.model_config.struct_dim:].cpu()
            
            new_graph.struct_feat = torch.cat([
                original_graph.struct_feat.cpu(),
                new_struct
            ], dim=0)
            
            new_graph.seq_feat = torch.cat([
                original_graph.seq_feat.cpu(),
                new_seq
            ], dim=0)
        
        # 确定新边
        new_edges = self.connector.determine_all_connections(
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
        
        # 更新节点标签（所有新节点都是结合位点）
        new_labels = torch.ones(n_new, dtype=torch.long)
        new_graph.y = torch.cat([
            original_graph.y.cpu(),
            new_labels
        ])
        
        # 处理位置信息
        if hasattr(original_graph, 'pos'):
            pos_list = []
            for info in generation_info:
                anchor = info['anchor']
                pos_list.append(original_graph.pos[anchor].cpu())
            new_pos = torch.stack(pos_list)
            new_graph.pos = torch.cat([
                original_graph.pos.cpu(),
                new_pos
            ], dim=0)
        
        new_graph.num_nodes = n_old + n_new
        
        return new_graph
    
    def augment_single_graph(self, protein_graph) -> object:
        """
        增强单个蛋白质图
        """
        n_needed = self.calculate_needed_binding_sites(protein_graph)
        
        if n_needed <= 0:
            print("  已达到目标比例，无需增强")
            return protein_graph
        
        print(f"  正在生成 {n_needed} 个结合位点...")
        new_features, new_info = self.generate_binding_sites(
            protein_graph,
            n_needed
        )
        
        augmented_graph = self.add_binding_sites_to_graph(
            protein_graph,
            new_features,
            new_info
        )
        
        aug_counts = torch.bincount(augmented_graph.y, minlength=2)
        print(f"  结果: 非结合位点 {aug_counts[0]}, 结合位点 {aug_counts[1]}")
        print(f"  新比例: {aug_counts[1]/aug_counts[0]:.3f}")
        
        return augmented_graph
    
    def augment_multiple_graphs(self,
                               graphs: List,
                               output_path: Optional[str] = None) -> List:
        """
        批量增强多个蛋白质图
        """
        print(f"\n开始批量增强 {len(graphs)} 个蛋白质图")
        
        augmented_graphs = []
        total_before = 0
        total_after = 0
        total_generated = 0
        
        for idx, graph in enumerate(tqdm(graphs, desc="增强图中")):
            before_counts = torch.bincount(graph.y, minlength=2)
            total_before += before_counts[1].item()
            
            augmented = self.augment_single_graph(graph)
            augmented_graphs.append(augmented)
            
            after_counts = torch.bincount(augmented.y, minlength=2)
            total_after += after_counts[1].item()
            total_generated += (after_counts[1] - before_counts[1]).item()
        
        print("\n" + "="*50)
        print("增强总体统计")
        print("="*50)
        print(f"处理图数量: {len(graphs)}")
        print(f"结合位点总数:")
        print(f"  增强前: {total_before}")
        print(f"  增强后: {total_after}")
        print(f"  新增: {total_generated}")
        print(f"平均每图新增: {total_generated/len(graphs):.1f} 个结合位点")
        
        if output_path:
            torch.save(augmented_graphs, output_path)
            print(f"\n增强后的图已保存到: {output_path}")
        
        return augmented_graphs


def main():
    """主函数 - 直接在代码中配置参数"""
    
    # 设置随机种子
    set_seed(SEED)
    
    # 创建配置
    model_config = ModelConfig(
        node_feature_dim=1408,
        n_classes=2
    )
    
    aug_config = AugmentationConfig(
        target_ratio=TARGET_RATIO,
        connection_strategy=CONNECTION_STRATEGY,
        similarity_threshold=SIMILARITY_THRESHOLD
    )
    
    # 创建增强器
    augmentor = ProteinGraphAugmentor(
        model_path=MODEL_PATH,
        model_config=model_config,
        aug_config=aug_config,
        device=DEVICE
    )
    
    # 加载新蛋白质图
    print(f"\n加载新蛋白质图: {INPUT_PATH}")
    new_graphs = torch.load(INPUT_PATH)
    print(f"加载了 {len(new_graphs)} 个蛋白质图")
    
    # 批量增强
    augmentor.augment_multiple_graphs(
        graphs=new_graphs,
        output_path=OUTPUT_PATH
    )


if __name__ == "__main__":
    main()