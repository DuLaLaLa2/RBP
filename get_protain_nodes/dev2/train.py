"""
训练脚本 - 训练GAN模型
直接在代码中配置参数，无需命令行
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
import random

from config import ModelConfig, TrainingConfig
from models import (
    TransformerProteinGenerator,
    TransformerProteinDiscriminator,
    GraphContextExtractor
)
from utils import (
    ProteinGraphDataset,
    LossHistory,
    ModelCheckpoint,
    set_seed
)


# ==================== 在这里直接配置训练参数 ====================

# 数据路径配置
INPUT_PATH = r"get_graph_data\\data\\pyg_graph_datas_495_train.pt"  # 输入：495个蛋白质图文件路径
OUTPUT_DIR = r"get_protain_nodes\\model"                         # 模型保存目录

# 模型配置
NODE_FEATURE_DIM = 1408    # 节点特征维度 (128结构 + 1280序列)
N_CLASSES = 2               # 类别数：0(非结合位点), 1(结合位点)
MINORITY_CLASSES = [1]      # 少数类：结合位点

# 训练配置
EPOCHS = 100                # 训练轮数
BATCH_SIZE = 1              # 批次大小（每个蛋白质图作为一个batch）
LEARNING_RATE = 1e-5        # 学习率
SEED = 42                   # 随机种子
DEVICE = 'cpu'              # 设备：'cuda' 或 'cpu'

# 恢复训练配置（如果需要从检查点恢复）
RESUME_PATH = None          # 例如：'./checkpoints/latest.pt' 或 None

# ==============================================================


class GANTrainer:
    """
    GAN训练器
    
    功能：训练生成器和判别器，重点关注结合位点（类别1）的生成
    """
    
    def __init__(self,
                 model_config: ModelConfig,
                 training_config: TrainingConfig,
                 minority_classes: list):
        
        self.model_config = model_config
        self.training_config = training_config
        self.minority_classes = minority_classes
        
        # 设置设备
        self.device = torch.device(training_config.device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化模型
        self.generator = TransformerProteinGenerator(model_config).to(self.device)
        self.discriminator = TransformerProteinDiscriminator(model_config).to(self.device)
        self.context_extractor = GraphContextExtractor(model_config).to(self.device)
        
        # 检查模型初始权重
        for name, param in self.generator.named_parameters():
            if torch.isnan(param).any():
                print(f"警告: 生成器参数 {name} 包含NaN")
        
        # 打印模型参数量
        self._print_model_sizes()
        
        # 初始化优化器
        self.g_optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=training_config.learning_rate,
            betas=(training_config.beta1, training_config.beta2),
            weight_decay=training_config.weight_decay
        )
        
        self.d_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=training_config.learning_rate,
            betas=(training_config.beta1, training_config.beta2),
            weight_decay=training_config.weight_decay
        )
        
        self.c_optimizer = torch.optim.AdamW(
            self.context_extractor.parameters(),
            lr=training_config.learning_rate,
            betas=(training_config.beta1, training_config.beta2),
            weight_decay=training_config.weight_decay
        )
        
        # 学习率调度器
        self.g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.g_optimizer, T_max=training_config.epochs
        )
        self.d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.d_optimizer, T_max=training_config.epochs
        )
        self.c_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.c_optimizer, T_max=training_config.epochs
        )
        
        # 损失函数
        self.criterion = nn.BCELoss()
        
        # 损失历史
        self.loss_history = LossHistory()
        
        # 检查点管理器
        self.checkpointer = ModelCheckpoint(save_dir=training_config.output_dir)
    
    def _print_model_sizes(self):
        """打印模型参数量"""
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        c_params = sum(p.numel() for p in self.context_extractor.parameters())
        
        print(f"\n模型参数量:")
        print(f"  生成器: {g_params/1e6:.2f}M")
        print(f"  判别器: {d_params/1e6:.2f}M")
        print(f"  上下文提取器: {c_params/1e6:.2f}M")
        print(f"  总计: {(g_params+d_params+c_params)/1e6:.2f}M\n")
    
    def train_step(self, real_graph):
        """
        单步训练 - 专门针对结合位点增强
        """
        real_graph = real_graph.to(self.device)
        
        # 获取节点特征和标签
        all_nodes = real_graph.x
        all_labels = real_graph.y
        
        # 调试：检查标签
        unique_labels = torch.unique(all_labels)
        if unique_labels.numel() == 0 or unique_labels.max() > 1:
            print(f"警告: 标签异常 unique_labels={unique_labels}, max={all_labels.max()}")
        
        # 分离多数类(0)和少数类(1)
        majority_nodes = all_nodes[all_labels == 0]
        minority_nodes = all_nodes[all_labels == 1]
        
        majority_count = len(majority_nodes)
        minority_count = len(minority_nodes)
        
        # 提取图上下文
        graph_context = self.context_extractor(real_graph)
        
        # 检查context有效性
        if torch.isnan(graph_context).any() or torch.isinf(graph_context).any():
            print("警告: graph_context包含NaN/Inf")
            return 0.0, 0.0
        
        # ==================== 训练判别器 ====================
        self.d_optimizer.zero_grad()
        
        # 1. 真实节点损失
        real_output = self.discriminator(all_nodes.unsqueeze(0))
        d_real_loss = self.criterion(real_output, torch.ones_like(real_output))
        
        # 2. 生成假节点用于训练
        fake_nodes_list = []
        
        # 生成结合位点（少数类）- 用于训练
        if minority_count > 0:
            # 训练时生成适量假节点，不是最终增强的数量
            n_fake_minority = max(minority_count * 2, 32)
            noise = torch.randn(n_fake_minority, self.model_config.latent_dim).to(self.device)
            class_cond = F.one_hot(
                torch.tensor([1] * n_fake_minority),
                num_classes=2
            ).float().to(self.device)
            context_batch = graph_context.repeat(n_fake_minority, 1)
            
            # 检查输入有效性
            if torch.isnan(noise).any() or torch.isinf(noise).any():
                print("警告: noise包含NaN/Inf")
                return 0.0, 0.0
            if torch.isnan(context_batch).any() or torch.isinf(context_batch).any():
                print("警告: context_batch包含NaN/Inf")
                return 0.0, 0.0
            
            fake_minority = self.generator(noise, class_cond, context_batch, n_generate=1)
            fake_nodes_list.append(fake_minority.squeeze(1))
        
        # 也生成一些非结合位点，保持判别器对两类都有判断力
        if majority_count > 0:
            n_fake_majority = min(majority_count // 4, 50)
            if n_fake_majority > 0:
                noise = torch.randn(n_fake_majority, self.model_config.latent_dim).to(self.device)
                class_cond = F.one_hot(
                    torch.tensor([0] * n_fake_majority),
                    num_classes=2
                ).float().to(self.device)
                context_batch = graph_context.repeat(n_fake_majority, 1)
                
                fake_majority = self.generator(noise, class_cond, context_batch, n_generate=1)
                fake_nodes_list.append(fake_majority.squeeze(1))
        
        if fake_nodes_list:
            fake_nodes = torch.cat(fake_nodes_list, dim=0)
            
            # 检查数值有效性
            if torch.isnan(fake_nodes).any() or torch.isinf(fake_nodes).any():
                print("警告: 生成节点包含NaN/Inf，跳过该batch")
                return 0.0, 0.0
            
            fake_output = self.discriminator(fake_nodes.unsqueeze(0))
            
            # 确保输出在有效范围内
            fake_output = torch.clamp(fake_output, min=1e-7, max=1-1e-7)
            
            d_fake_loss = self.criterion(fake_output, torch.zeros_like(fake_output))
            d_loss = d_real_loss + d_fake_loss
        else:
            d_loss = d_real_loss
        
        # 检查d_loss有效性
        if torch.isnan(d_loss) or torch.isinf(d_loss):
            print("警告: d_loss为NaN/Inf")
            return 0.0, 0.0
        
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.discriminator.parameters(), 
            self.training_config.grad_clip
        )
        self.d_optimizer.step()
        
        # ==================== 训练生成器 ====================
        self.g_optimizer.zero_grad()
        
        # 生成结合位点（重点训练生成器生成好的结合位点）
        n_gen = max(minority_count * 2, 32)
        noise = torch.randn(n_gen, self.model_config.latent_dim).to(self.device)
        class_cond = F.one_hot(
            torch.tensor([1] * n_gen),
            num_classes=2
        ).float().to(self.device)
        context_batch = graph_context.repeat(n_gen, 1)
        
        fake_minority = self.generator(noise, class_cond, context_batch, n_generate=1)
        fake_minority = fake_minority.squeeze(1)
        
        fake_output = self.discriminator(fake_minority.unsqueeze(0))
        g_loss = self.criterion(fake_output, torch.ones_like(fake_output))
        
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(), 
            self.training_config.grad_clip
        )
        self.g_optimizer.step()
        
        return d_loss.item(), g_loss.item()
    
    def train(self, dataset: ProteinGraphDataset, resume_from: str = None):
        """
        完整训练循环
        """
        start_epoch = 0
        
        # 恢复训练
        if resume_from and os.path.exists(resume_from):
            checkpoint = self.checkpointer.load(
                resume_from,
                self.generator,
                self.discriminator,
                self.context_extractor,
                self.g_optimizer,
                self.d_optimizer,
                self.c_optimizer
            )
            start_epoch = checkpoint['epoch']
            self.loss_history.history = checkpoint.get('loss_history', {'d_loss': [], 'g_loss': [], 'epoch': []})
            print(f"从 epoch {start_epoch} 恢复训练")
        
        print(f"\n开始训练，共 {len(dataset)} 个蛋白质图，{self.training_config.epochs} 轮")
        print(f"少数类: {self.minority_classes}")
        
        for epoch in range(start_epoch, self.training_config.epochs):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            
            # 打乱数据顺序
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            
            pbar = tqdm(indices, desc=f"Epoch {epoch+1}/{self.training_config.epochs}")
            for idx in pbar:
                graph = dataset[idx]
                d_loss, g_loss = self.train_step(graph)
                
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss
                
                pbar.set_postfix({
                    'D': f'{d_loss:.4f}',
                    'G': f'{g_loss:.4f}'
                })
            
            # 计算平均损失
            avg_d_loss = epoch_d_loss / len(dataset)
            avg_g_loss = epoch_g_loss / len(dataset)
            
            # 更新学习率
            self.g_scheduler.step()
            self.d_scheduler.step()
            self.c_scheduler.step()
            
            # 记录损失
            self.loss_history.update(epoch + 1, avg_d_loss, avg_g_loss)
            
            # 打印进度
            print(f"\nEpoch {epoch+1}/{self.training_config.epochs} - "
                  f"D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}, "
                  f"LR: {self.g_scheduler.get_last_lr()[0]:.6f}")
            
            # 保存检查点
            if (epoch + 1) % self.training_config.save_interval == 0:
                is_best = avg_g_loss < self.loss_history.get_average().get('g_loss', float('inf'))
                
                self.checkpointer.save(
                    epoch=epoch + 1,
                    generator=self.generator,
                    discriminator=self.discriminator,
                    context_extractor=self.context_extractor,
                    g_optimizer=self.g_optimizer,
                    d_optimizer=self.d_optimizer,
                    c_optimizer=self.c_optimizer,
                    config={
                        'model_config': vars(self.model_config),
                        'training_config': vars(self.training_config),
                        'minority_classes': self.minority_classes
                    },
                    loss_history=self.loss_history.history,
                    is_best=is_best
                )
        
        print("\n训练完成!")
        
        # 保存最终模型
        self.checkpointer.save(
            epoch=self.training_config.epochs,
            generator=self.generator,
            discriminator=self.discriminator,
            context_extractor=self.context_extractor,
            g_optimizer=self.g_optimizer,
            d_optimizer=self.d_optimizer,
            c_optimizer=self.c_optimizer,
            config={
                'model_config': vars(self.model_config),
                'training_config': vars(self.training_config),
                'minority_classes': self.minority_classes
            },
            loss_history=self.loss_history.history,
            is_best=False
        )
        
        # 保存损失历史
        self.loss_history.save(os.path.join(self.checkpointer.save_dir, 'loss_history.json'))


def main():
    """主函数 - 直接在代码中配置参数"""
    
    # 设置随机种子
    set_seed(SEED)
    
    # 创建配置
    model_config = ModelConfig(
        node_feature_dim=NODE_FEATURE_DIM,
        n_classes=N_CLASSES
    )
    
    training_config = TrainingConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        output_dir=OUTPUT_DIR  
    )
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载数据集
    print(f"\n{'='*50}")
    print("加载数据集")
    print(f"{'='*50}")
    print(f"数据路径: {INPUT_PATH}")
    
    dataset = ProteinGraphDataset(
        INPUT_PATH, 
        model_config,
        minority_classes=MINORITY_CLASSES
    )
    
    # 检查数据是否有NaN
    print("\n检查数据质量...")
    for i, graph in enumerate(dataset.graphs[:5]):  # 检查前5个图
        if torch.isnan(graph.x).any():
            print(f"警告: 图 {i} 包含NaN特征!")
        if torch.isinf(graph.x).any():
            print(f"警告: 图 {i} 包含Inf特征!")
    
    # 创建训练器
    print(f"\n{'='*50}")
    print("初始化训练器")
    print(f"{'='*50}")
    trainer = GANTrainer(
        model_config=model_config,
        training_config=training_config,
        minority_classes=MINORITY_CLASSES
    )
    
    # 开始训练
    print(f"\n{'='*50}")
    print("开始训练")
    print(f"{'='*50}")
    trainer.train(dataset, resume_from=RESUME_PATH)
    
    print(f"\n训练完成！模型保存在 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()