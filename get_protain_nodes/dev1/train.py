"""
训练脚本 - 只负责训练GAN模型
protein_gan_models -> train -> augment
"""

import torch
import random
from torch.utils.data import Dataset
from tqdm import tqdm
import argparse

from protein_gan_models import (
    TransformerProteinGenerator,
    TransformerProteinDiscriminator,
    GraphContextExtractor,
    set_seed
)


class ProteinGraphDataset(Dataset):
    """蛋白质图数据集"""
    def __init__(self, graph_path: str):
        self.graphs = torch.load(graph_path)
        print(f"加载了 {len(self.graphs)} 个蛋白质图")
        
        # 检查数据格式
        for i, graph in enumerate(self.graphs):
            # 你的数据已经有y作为节点标签，不需要额外创建
            if not hasattr(graph, 'y'):
                print(f"警告: 图 {i} 没有节点标签 y")
            else:
                # 确保y是long类型
                graph.y = graph.y.long()
            
            # 检查特征维度
            if graph.x.size(1) != 1408:
                print(f"警告: 图 {i} 的特征维度是 {graph.x.size(1)}，期望是1408")
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int):
        return self.graphs[idx]


class GANTrainer:
    """GAN训练器"""
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
        """单步训练"""
        real_graph = real_graph.to(self.device)
        
        real_nodes = real_graph.x.unsqueeze(0)
        batch_size = 1
        num_nodes = real_nodes.size(1)
        
        if hasattr(real_graph, 'y'):
            # 随机选择一个节点的标签作为图标签（或者你可以用多数投票）
            class_label = real_graph.y[0].item()
            real_labels = F.one_hot(
                torch.tensor([class_label]), 
                num_classes=self.n_classes
            ).float().to(self.device)
        else:
            # 如果没有标签，使用零向量
            real_labels = torch.zeros(1, self.n_classes).to(self.device)
            print("警告: 图没有标签 y")
        
        # 提取图上下文
        context = self.context_extractor(real_graph)
        
        # 训练判别器
        self.d_optimizer.zero_grad()
        
        real_output = self.discriminator(real_nodes)
        d_real_loss = self.criterion(real_output, torch.ones_like(real_output))
        
        noise = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
        fake_nodes = self.generator(noise, real_labels, context, n_generate=num_nodes)
        fake_output = self.discriminator(fake_nodes.detach())
        d_fake_loss = self.criterion(fake_output, torch.zeros_like(fake_output))
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.d_optimizer.step()
        
        # 训练生成器
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
        """完整训练循环"""
        print(f"开始训练，共 {len(dataset)} 个蛋白质图，{epochs} 轮")
        
        best_g_loss = float('inf')
        
        for epoch in range(epochs):
            total_d_loss = 0
            total_g_loss = 0
            
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
            
            # 保存最佳模型
            if avg_g_loss < best_g_loss:
                best_g_loss = avg_g_loss
                self.save_model(save_path.replace('.pt', '_best.pt'))
                print(f"保存最佳模型，G_loss: {best_g_loss:.4f}")
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_model(f"{save_path}_epoch{epoch+1}.pt")
        
        # 保存最终模型
        self.save_model(save_path)
        print(f"训练完成，模型已保存到 {save_path}")
    
    def save_model(self, path: str):
        """保存完整模型"""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'context_extractor': self.context_extractor.state_dict(),
            'model_config': {
                'node_feature_dim': self.node_feature_dim,
                'n_classes': self.n_classes,
                'latent_dim': self.generator.latent_dim
            }
        }, path)
        print(f"模型已保存到 {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.context_extractor.load_state_dict(checkpoint['context_extractor'])
        print(f"模型已从 {path} 加载")
        return checkpoint.get('model_config', {})


def main():
    parser = argparse.ArgumentParser(description='训练蛋白质图GAN模型')
    parser.add_argument('--input_path', type=str, required=True,
                        help='训练数据路径 (495个蛋白质图的.pt文件)')
    parser.add_argument('--output_model', type=str, default='protein_gan_model.pt',
                        help='输出的模型文件路径')
    parser.add_argument('--node_feature_dim', type=int, default=1280,
                        help='节点特征维度')
    parser.add_argument('--n_classes', type=int, default=10,
                        help='类别数')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
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
    
    # 加载数据集
    print(f"\n=== 加载数据集 ===")
    dataset = ProteinGraphDataset(args.input_path)
    
    # 创建训练器
    print(f"\n=== 初始化GAN ===")
    trainer = GANTrainer(
        node_feature_dim=args.node_feature_dim,
        n_classes=args.n_classes,
        device=device,
        lr=args.lr
    )
    
    # 开始训练
    print(f"\n=== 开始训练 ===")
    trainer.train(dataset, epochs=args.epochs, save_path=args.output_model)
    
    print(f"\n=== 训练完成！ ===")
    print(f"模型已保存到: {args.output_model}")


if __name__ == "__main__":
    import torch.nn as nn
    import torch.nn.functional as F
    main()