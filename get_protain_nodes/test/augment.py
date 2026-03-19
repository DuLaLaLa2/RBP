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
from edge_predictor import EdgePredictor


MODEL_PATH = r"get_protain_nodes\\test\\model\\latest.pt"
EDGE_PREDICTOR_PATH = r"get_protain_nodes\\test\\model\\edge_predictor.pt"
INPUT_PATH = r"get_protain_nodes\\test\\data\\train_graphs_for_gan.pt"
OUTPUT_PATH = r"get_protain_nodes\\test\\data\\enhanced_train_graphs.pt"

TARGET_RATIO = 0.2
EDGE_THRESHOLD = 0.5
MAX_EDGES_RATIO = 0.3

SEED = 42
DEVICE = 'cuda'


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class ProteinGraphAugmentor:
    def __init__(self, model_path: str, edge_predictor_path: str, model_config: ModelConfig,
                 aug_config: AugmentationConfig, device: str = 'cuda'):

        self.model_config = model_config
        self.aug_config = aug_config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self._load_model(model_path)
        self._load_edge_predictor(edge_predictor_path)
        
        self.connector = SyntheticNodeConnector(
            edge_predictor=self.edge_predictor,
            threshold=EDGE_THRESHOLD,
            max_edges_ratio=MAX_EDGES_RATIO
        )

        print(f"\n{'='*50}")
        print("增强器配置")
        print(f"{'='*50}")
        print(f"目标比例: 少数类/多数类 = {aug_config.target_ratio}")
        print(f"边预测阈值: {EDGE_THRESHOLD}")
        print(f"最大边比例: {MAX_EDGES_RATIO}")
        print(f"设备: {self.device}")

    def _load_model(self, model_path: str):
        print(f"加载GAN模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.generator = TransformerProteinGenerator(self.model_config).to(self.device)
        self.discriminator = TransformerProteinDiscriminator(self.model_config).to(self.device)
        self.context_extractor = GraphContextExtractor(self.model_config).to(self.device)

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.context_extractor.load_state_dict(checkpoint['context_extractor_state_dict'])

        self.generator.eval()
        self.discriminator.eval()
        self.context_extractor.eval()

        print("GAN模型加载成功！")
    
    def _load_edge_predictor(self, path: str):
        print(f"加载边预测器: {path}")
        self.edge_predictor = EdgePredictor(
            node_feature_dim=self.model_config.node_feature_dim,
            hidden_dim=256
        ).to(self.device)
        self.edge_predictor.load_state_dict(torch.load(path, map_location=self.device))
        self.edge_predictor.eval()
        print("边预测器加载成功！")

    def generate_nodes(self, protein_graph, n_nodes: int) -> torch.Tensor:
        context = self.context_extractor(protein_graph.to(self.device))

        class_cond = F.one_hot(
            torch.tensor([1]),
            num_classes=2
        ).float().to(self.device)

        synthetic_features = []
        batch_size = self.aug_config.generation_batch_size

        with torch.no_grad():
            for i in range(0, n_nodes, batch_size):
                current_batch = min(batch_size, n_nodes - i)

                noise = torch.randn(current_batch, self.model_config.latent_dim).to(self.device)
                class_batch = class_cond.repeat(current_batch, 1)
                context_batch = context.repeat(current_batch, 1)

                nodes = self.generator(noise, class_batch, context_batch, n_generate=1)

                synthetic_features.append(nodes.squeeze(1).cpu())

        return torch.cat(synthetic_features, dim=0)

    def calculate_needed_nodes(self, protein_graph) -> int:
        class_counts = torch.bincount(protein_graph.y, minlength=2)
        non_binding = class_counts[0].item()
        binding = class_counts[1].item()

        if non_binding == 0:
            return 0

        target_binding = int(non_binding * self.aug_config.target_ratio)
        needed = max(0, target_binding - binding)

        return needed

    def augment_single_graph(self, original_graph) -> object:
        n_needed = self.calculate_needed_nodes(original_graph)

        if n_needed <= 0:
            return original_graph

        print(f"  正在生成 {n_needed} 个结合位点...")

        new_features = self.generate_nodes(original_graph, n_needed)

        n_old = original_graph.num_nodes
        n_new = len(new_features)

        original_x = original_graph.x.cpu()
        original_y = original_graph.y.cpu()
        original_edge = original_graph.edge_index.cpu() if original_graph.edge_index is not None else None
        
        original_x = original_x.to(self.device)
        new_features = new_features.to(self.device)

        new_edges = self.connector.determine_connections(
            original_x,
            new_features
        ).cpu()

        if original_edge is not None:
            combined_edge = torch.cat([original_edge, new_edges], dim=1)
        else:
            combined_edge = new_edges

        if hasattr(original_graph, 'clone'):
            new_graph = original_graph.clone()
            new_graph.x = torch.cat([original_x, new_features], dim=0)
            new_graph.y = torch.cat([original_y, torch.ones(n_new, dtype=torch.long)], dim=0)
            new_graph.edge_index = combined_edge
        else:
            new_graph = type(original_graph)()
            new_graph.x = torch.cat([original_x, new_features], dim=0)
            new_graph.y = torch.cat([original_y, torch.ones(n_new, dtype=torch.long)], dim=0)
            new_graph.edge_index = combined_edge

        if hasattr(original_graph, 'pos'):
            new_graph.pos = original_graph.pos.cpu()

        new_graph.num_nodes = n_old + n_new

        aug_counts = torch.bincount(new_graph.y, minlength=2)
        print(f"  结果: 类别0={aug_counts[0]}, 类别1={aug_counts[1]}")
        print(f"  新比例: {aug_counts[1]/aug_counts[0]:.3f}")

        return new_graph

    def augment_multiple_graphs(self, graphs: List, output_path: Optional[str] = None) -> List:
        print(f"\n开始批量增强 {len(graphs)} 个蛋白质图")

        augmented_graphs = []
        total_before = 0
        total_after = 0

        for idx, graph in enumerate(tqdm(graphs, desc="增强图中")):
            before_counts = torch.bincount(graph.y, minlength=2)
            total_before += before_counts[1].item()

            augmented = self.augment_single_graph(graph)
            augmented_graphs.append(augmented)

            after_counts = torch.bincount(augmented.y, minlength=2)
            total_after += after_counts[1].item()

        print("\n" + "="*50)
        print("增强总体统计")
        print("="*50)
        print(f"处理图数量: {len(graphs)}")
        print(f"结合位点:")
        print(f"  增强前: {total_before}")
        print(f"  增强后: {total_after}")
        print(f"  新增: {total_after - total_before}")
        print(f"平均每图新增: {(total_after - total_before)/len(graphs):.1f} 个结合位点")

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 确保所有张量在CPU上
            for g in augmented_graphs:
                g.x = g.x.cpu()
                g.y = g.y.cpu()
                if g.edge_index is not None:
                    g.edge_index = g.edge_index.cpu()
                if hasattr(g, 'pos') and g.pos is not None:
                    g.pos = g.pos.cpu()
            
            torch.save(augmented_graphs, output_path)
            print(f"\n增强后的图已保存到: {output_path}")

        return augmented_graphs


def main():
    set_seed(SEED)

    model_config = ModelConfig(
        node_feature_dim=1408,
        n_classes=2
    )

    aug_config = AugmentationConfig(
        target_ratio=TARGET_RATIO
    )

    augmentor = ProteinGraphAugmentor(
        model_path=MODEL_PATH,
        edge_predictor_path=EDGE_PREDICTOR_PATH,
        model_config=model_config,
        aug_config=aug_config,
        device=DEVICE
    )

    print(f"\n加载蛋白质图: {INPUT_PATH}")
    graphs = torch.load(INPUT_PATH, weights_only=False)
    print(f"加载了 {len(graphs)} 个蛋白质图")

    augmentor.augment_multiple_graphs(
        graphs=graphs,
        output_path=OUTPUT_PATH
    )


if __name__ == "__main__":
    main()