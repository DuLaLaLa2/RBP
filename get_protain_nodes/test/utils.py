import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
import os


class ProteinGraphDataset(Dataset):
    def __init__(self, graph_path: str, config, minority_classes: Optional[List[int]] = None):
        self.graphs = torch.load(graph_path, weights_only=False)
        self.config = config
        self.minority_classes = minority_classes or []

        print(f"加载了 {len(self.graphs)} 个蛋白质图")
        self._validate()
        self._print_stats()

    def _validate(self):
        for i, graph in enumerate(self.graphs):
            required_attrs = ['x', 'edge_index', 'y', 'num_nodes']
            for attr in required_attrs:
                if not hasattr(graph, attr):
                    raise ValueError(f"图 {i} 缺少必要属性: {attr}")
            graph.y = graph.y.long()

    def _print_stats(self):
        total_nodes = 0
        class_counts = {}
        for graph in self.graphs:
            labels = graph.y.tolist()
            total_nodes += len(labels)
            for label in labels:
                class_counts[label] = class_counts.get(label, 0) + 1

        print(f"\n数据集统计:")
        print(f"总节点数: {total_nodes}")
        for label, count in sorted(class_counts.items()):
            ratio = count / total_nodes
            mark = " (少数类)" if label in self.minority_classes else ""
            print(f"  类别 {label}: {count} ({ratio:.2%}){mark}")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


class LossHistory:
    def __init__(self):
        self.history = {'d_loss': [], 'g_loss': [], 'epoch': []}

    def update(self, epoch, d_loss, g_loss):
        self.history['epoch'].append(epoch)
        self.history['d_loss'].append(d_loss)
        self.history['g_loss'].append(g_loss)

    def get_average(self):
        if not self.history['d_loss']:
            return {'d_loss': 0, 'g_loss': 0}
        return {
            'd_loss': sum(self.history['d_loss']) / len(self.history['d_loss']),
            'g_loss': sum(self.history['g_loss']) / len(self.history['g_loss'])
        }

    def save(self, path):
        import json
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


class ModelCheckpoint:
    def __init__(self, save_dir: str = './checkpoints'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(self, epoch, generator, discriminator, context_extractor,
             g_optimizer, d_optimizer, c_optimizer, config, loss_history, is_best=False):
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

        latest_path = os.path.join(self.save_dir, 'latest.pt')
        torch.save(checkpoint, latest_path)

        epoch_path = os.path.join(self.save_dir, f'epoch_{epoch}.pt')
        torch.save(checkpoint, epoch_path)

        if is_best:
            best_path = os.path.join(self.save_dir, 'best.pt')
            torch.save(checkpoint, best_path)

        print(f"检查点已保存: epoch {epoch}")

    def load(self, path, generator, discriminator, context_extractor,
             g_optimizer, d_optimizer, c_optimizer):
        checkpoint = torch.load(path, map_location='cpu')
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        context_extractor.load_state_dict(checkpoint['context_extractor_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        c_optimizer.load_state_dict(checkpoint['c_optimizer_state_dict'])
        return checkpoint


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False