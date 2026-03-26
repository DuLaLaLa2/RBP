##################################################################################################################
# 基于EGNN的图分类完整实现（最优版）
# 包含：特征更新、分类头、训练/验证/测试流程、优化策略、早停、混合精度等
##################################################################################################################
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# ===================== 核心工具函数（优化版） =====================
def unsorted_segment_sum(data, segment_ids, num_segments):
    """优化的无排序段求和，支持批量处理"""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0.0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    """优化的无排序段求平均，防止除零"""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0.0)
    count = data.new_full(result_shape, 0.0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1.0)

def get_edges(n_nodes):
    """生成全连接边（排除自环）"""
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    return [rows, cols]

def get_edges_batch(n_nodes, batch_size):
    """批量生成边索引，优化内存占用"""
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1, dtype=torch.float32)
    
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr

# ===================== E_GCL层（保留核心并优化） =====================
class E_GCL(nn.Module):
    """E(n)等变卷积层（优化初始化和数值稳定性）"""
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, 
                 act_fn=nn.SiLU(), residual=True, attention=False, 
                 normalize=False, coords_agg='mean', tanh=False):
        super().__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        # 边MLP（优化初始化）
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.LayerNorm(hidden_nf),  # 增加层归一化提升稳定性
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.LayerNorm(hidden_nf)
        )
        # 节点MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.LayerNorm(hidden_nf),
            nn.Linear(hidden_nf, output_nf)
        )
        # 坐标MLP（优化初始化）
        coord_mlp = [
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.LayerNorm(hidden_nf),
            nn.Linear(hidden_nf, 1, bias=False)
        ]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)
        
        # 注意力分支
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid()
            )
        
        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """修复后的权重初始化，无bug"""
        if isinstance(m, nn.Linear):
            # 专门给坐标输出层设置小权重
            if hasattr(m, 'out_features') and m.out_features == 1:
                torch.nn.init.xavier_uniform_(m.weight, gain=0.001)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

    def edge_model(self, source, target, radial, edge_attr):
        """边特征更新（优化拼接逻辑）"""
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        """节点特征更新"""
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out  # 残差连接
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        """坐标更新（等变核心）"""
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise ValueError(f"Invalid coords_agg: {self.coords_agg}")
        
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        """坐标转径向特征（优化数值稳定性）"""
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1, keepdim=True)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        """前向传播（特征+坐标联合更新）"""
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[edge_index[0]], h[edge_index[1]], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr

# ===================== EGNN主干网络（增强版） =====================
class EGNN(nn.Module):
    """EGNN主干网络（支持特征逐层更新）"""
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, 
                 device='cuda' if torch.cuda.is_available() else 'cpu', 
                 act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, 
                 normalize=False, tanh=False):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        
        # 输入/输出嵌入（优化维度匹配）
        self.embedding_in = nn.Sequential(
            nn.Linear(in_node_nf, hidden_nf),
            nn.LayerNorm(hidden_nf),
            act_fn
        )
        self.embedding_out = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf // 2),
            act_fn,
            nn.LayerNorm(hidden_nf // 2),
            nn.Linear(hidden_nf // 2, out_node_nf)
        )
        
        # 堆叠EGNN层
        self.gcl_layers = nn.ModuleList([
            E_GCL(hidden_nf, hidden_nf, hidden_nf, edges_in_d=in_edge_nf,
                  act_fn=act_fn, residual=residual, attention=attention,
                  normalize=normalize, tanh=tanh)
            for _ in range(n_layers)
        ])
        
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr):
        """
        前向传播：逐层更新节点特征和坐标
        Args:
            h: 节点特征 [N_total, in_node_nf]
            x: 节点坐标 [N_total, 3]
            edges: 边索引 [2, E_total]
            edge_attr: 边特征 [E_total, in_edge_nf]
        Returns:
            h_out: 最终节点特征 [N_total, out_node_nf]
            x_out: 最终坐标 [N_total, 3]
            h_list: 各层特征（用于分析）[n_layers+1, N_total, hidden_nf]
        """
        h = self.embedding_in(h)
        h_list = [h.clone()]  # 记录初始特征
        
        # 逐层更新特征和坐标
        for idx, gcl in enumerate(self.gcl_layers):
            h, x, _ = gcl(h, edges, x, edge_attr)
            h_list.append(h.clone())
        
        h_out = self.embedding_out(h)
        return h_out, x, h_list

# ===================== 图分类头（最优设计） =====================
class EGNNClassifier(nn.Module):
    """EGNN分类器（包含图级聚合+分类头）"""
    def __init__(self, in_node_nf, hidden_nf, num_classes, in_edge_nf=0, 
                 n_layers=4, device='cuda', attention=True, normalize=True, 
                 tanh=False, dropout=0.1):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        
        # EGNN主干（输出节点级特征）
        self.egnn_backbone = EGNN(
            in_node_nf=in_node_nf,
            hidden_nf=hidden_nf,
            out_node_nf=hidden_nf,  # 输出节点特征维度
            in_edge_nf=in_edge_nf,
            device=device,
            n_layers=n_layers,
            attention=attention,
            normalize=normalize,
            tanh=tanh
        )
        
        # 图级聚合（多种聚合方式融合，最优实践）
        self.pooling = nn.Sequential(
            nn.Linear(hidden_nf * 3, hidden_nf),  # mean+max+sum 融合
            nn.SiLU(),
            nn.LayerNorm(hidden_nf),
            self.dropout
        )
        
        # 分类头（两层MLP+dropout，防止过拟合）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf // 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_nf // 2),
            self.dropout,
            nn.Linear(hidden_nf // 2, num_classes)
        )

    def graph_pooling(self, node_feat, batch_idx):
        """
        图级聚合：融合mean/max/sum三种聚合方式（最优聚合策略）
        Args:
            node_feat: 节点特征 [N_total, hidden_nf]
            batch_idx: 批次索引 [N_total,] （标记每个节点属于哪个图）
        Returns:
            graph_feat: 图级特征 [batch_size, hidden_nf]
        """
        batch_size = batch_idx.max() + 1
        
        # 三种聚合方式
        mean_feat = unsorted_segment_mean(node_feat, batch_idx, batch_size)
        sum_feat = unsorted_segment_sum(node_feat, batch_idx, batch_size)
        max_feat = torch.stack([
            node_feat[batch_idx == i].max(dim=0)[0] for i in range(batch_size)
        ], dim=0)
        
        # 融合聚合特征
        combined_feat = torch.cat([mean_feat, max_feat, sum_feat], dim=1)
        graph_feat = self.pooling(combined_feat)
        return graph_feat

    def forward(self, h, x, edges, edge_attr, batch_idx):
        """
        完整前向：节点特征更新 → 图聚合 → 分类
        Args:
            h: 节点特征 [N_total, in_node_nf]
            x: 节点坐标 [N_total, 3]
            edges: 边索引 [2, E_total]
            edge_attr: 边特征 [E_total, in_edge_nf]
            batch_idx: 批次索引 [N_total,]
        Returns:
            logits: 分类预测 [batch_size, num_classes]
            node_feat: 最终节点特征 [N_total, hidden_nf]
            graph_feat: 图级特征 [batch_size, hidden_nf]
        """
        # 1. EGNN特征更新
        node_feat, x_out, h_list = self.egnn_backbone(h, x, edges, edge_attr)
        
        # 2. 图级聚合
        graph_feat = self.graph_pooling(node_feat, batch_idx)
        
        # 3. 分类预测
        logits = self.classifier(graph_feat)
        
        return logits, node_feat, graph_feat

# ===================== 数据集封装（通用版） =====================
class GraphClassificationDataset(Dataset):
    """通用图分类数据集封装"""
    def __init__(self, num_samples=1000, n_nodes_range=(3, 10), n_feat=5, 
                 x_dim=3, num_classes=2, seed=42):
        """
        生成模拟数据（可替换为真实数据集加载逻辑）
        Args:
            num_samples: 样本数
            n_nodes_range: 每个图的节点数范围
            n_feat: 节点特征维度
            x_dim: 坐标维度
            num_classes: 分类类别数
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.data = []
        for _ in range(num_samples):
            n_nodes = np.random.randint(n_nodes_range[0], n_nodes_range[1]+1)
            
            # 节点特征
            h = torch.randn(n_nodes, n_feat, dtype=torch.float32)
            # 节点坐标
            x = torch.randn(n_nodes, x_dim, dtype=torch.float32)
            # 边索引和边特征
            edges, edge_attr = get_edges_batch(n_nodes, batch_size=1)
            edges = torch.stack(edges, dim=0)
            # 标签
            label = torch.randint(0, num_classes, (1,), dtype=torch.long)
            
            self.data.append({
                'h': h,
                'x': x,
                'edges': edges,
                'edge_attr': edge_attr,
                'label': label,
                'n_nodes': n_nodes
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """自定义collate函数，批量拼接图数据"""
    h_list, x_list, edges_list, edge_attr_list = [], [], [], []
    label_list, batch_idx_list = [], []
    n_nodes_cum = 0
    
    for item in batch:
        n_nodes = item['n_nodes']
        # 节点特征/坐标
        h_list.append(item['h'])
        x_list.append(item['x'])
        # 边索引（偏移）
        edges = item['edges'] + n_nodes_cum
        edges_list.append(edges)
        # 边特征
        edge_attr_list.append(item['edge_attr'])
        # 标签
        label_list.append(item['label'])
        # 批次索引
        batch_idx_list.append(torch.ones(n_nodes, dtype=torch.long) * len(label_list)-1)
        
        n_nodes_cum += n_nodes
    
    # 拼接所有数据
    h = torch.cat(h_list, dim=0)
    x = torch.cat(x_list, dim=0)
    edges = torch.cat(edges_list, dim=1)
    edge_attr = torch.cat(edge_attr_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    batch_idx = torch.cat(batch_idx_list, dim=0)
    
    return {
        'h': h,
        'x': x,
        'edges': edges,
        'edge_attr': edge_attr,
        'labels': labels,
        'batch_idx': batch_idx
    }

# ===================== 训练/验证/测试流程（最优策略） =====================
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, cfg):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = cfg
        
        # 设备
        self.device = cfg['device']
        # 损失函数
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        # 优化器（AdamW + 权重衰减，最优实践）
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg['lr'],
            weight_decay=cfg['weight_decay'],
            betas=(0.9, 0.999)
        )
        # 学习率调度器（余弦退火+早停）
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg['epochs'],
            eta_min=cfg['lr'] * 0.01
        )
        # 混合精度训练（加速+稳定）
        self.scaler = GradScaler() if cfg['fp16'] else None
        
        # 早停配置
        self.early_stop_patience = cfg['early_stop_patience']
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # 日志
        self.train_logs = {'loss': [], 'acc': []}
        self.val_logs = {'loss': [], 'acc': []}

    def compute_accuracy(self, logits, labels):
        """计算分类准确率"""
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        return acc

    def train_one_epoch(self, epoch):
        """单轮训练"""
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.cfg["epochs"]}')
        
        for batch in pbar:
            # 数据移到设备
            h = batch['h'].to(self.device)
            x = batch['x'].to(self.device)
            edges = batch['edges'].to(self.device)
            edge_attr = batch['edge_attr'].to(self.device)
            labels = batch['labels'].to(self.device)
            batch_idx = batch['batch_idx'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 混合精度前向
            if self.cfg['fp16']:
                with autocast():
                    logits, _, _ = self.model(h, x, edges, edge_attr, batch_idx)
                    loss = self.criterion(logits, labels)
                # 混合精度反向
                self.scaler.scale(loss).backward()
                # 梯度裁剪（防止梯度爆炸）
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, _, _ = self.model(h, x, edges, edge_attr, batch_idx)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # 计算指标
            acc = self.compute_accuracy(logits, labels)
            total_loss += loss.item()
            total_acc += acc.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc.item():.4f}'
            })
        
        # 平均指标
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_acc / len(self.train_loader)
        self.train_logs['loss'].append(avg_loss)
        self.train_logs['acc'].append(avg_acc)
        
        # 学习率调度
        self.scheduler.step()
        
        return avg_loss, avg_acc

    @torch.no_grad()
    def validate(self):
        """验证流程"""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        
        for batch in self.val_loader:
            # 数据移到设备
            h = batch['h'].to(self.device)
            x = batch['x'].to(self.device)
            edges = batch['edges'].to(self.device)
            edge_attr = batch['edge_attr'].to(self.device)
            labels = batch['labels'].to(self.device)
            batch_idx = batch['batch_idx'].to(self.device)
            
            # 前向传播
            logits, _, _ = self.model(h, x, edges, edge_attr, batch_idx)
            loss = self.criterion(logits, labels)
            
            # 计算指标
            acc = self.compute_accuracy(logits, labels)
            total_loss += loss.item()
            total_acc += acc.item()
        
        # 平均指标
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_acc / len(self.val_loader)
        self.val_logs['loss'].append(avg_loss)
        self.val_logs['acc'].append(avg_acc)
        
        return avg_loss, avg_acc

    @torch.no_grad()
    def test(self):
        """测试流程"""
        self.model.eval()
        total_acc = 0.0
        
        for batch in self.test_loader:
            # 数据移到设备
            h = batch['h'].to(self.device)
            x = batch['x'].to(self.device)
            edges = batch['edges'].to(self.device)
            edge_attr = batch['edge_attr'].to(self.device)
            labels = batch['labels'].to(self.device)
            batch_idx = batch['batch_idx'].to(self.device)
            
            # 前向传播
            logits, _, _ = self.model(h, x, edges, edge_attr, batch_idx)
            
            # 计算指标
            acc = self.compute_accuracy(logits, labels)
            total_acc += acc.item()
        
        avg_acc = total_acc / len(self.test_loader)
        return avg_acc

    def fit(self):
        """完整训练流程"""
        print(f"开始训练（设备：{self.device}）")
        for epoch in range(self.cfg['epochs']):
            # 训练
            train_loss, train_acc = self.train_one_epoch(epoch)
            # 验证
            val_loss, val_acc = self.validate()
            
            # 打印日志
            print(f'Epoch {epoch+1} | 训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f}')
            print(f'          | 验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}')
            print('-' * 50)
            
            # 早停逻辑
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                # 保存最优模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                }, self.cfg['ckpt_path'])
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop_patience:
                    print(f"早停触发（{self.early_stop_patience}轮无提升）")
                    break
        
        # 加载最优模型测试
        checkpoint = torch.load(self.cfg['ckpt_path'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        test_acc = self.test()
        print(f"\n测试准确率: {test_acc:.4f}")
        print(f"最优验证准确率: {self.best_val_acc:.4f}")
        
        return self.train_logs, self.val_logs, test_acc

# ===================== 主函数（一键运行） =====================
if __name__ == "__main__":
    # 配置参数（最优配置）
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_samples': 2000,          # 总样本数
        'n_nodes_range': (3, 10),     # 图节点数范围
        'n_feat': 8,                  # 节点特征维度
        'x_dim': 3,                   # 坐标维度
        'num_classes': 2,             # 分类类别数
        'hidden_nf': 64,              # EGNN隐藏层维度
        'n_layers': 4,                # EGNN层数
        'attention': True,            # 使用注意力
        'normalize': True,            # 坐标归一化
        'tanh': False,                # 坐标输出是否tanh
        'dropout': 0.1,               # dropout率
        'batch_size': 32,             # 批次大小
        'lr': 1e-3,                   # 学习率
        'weight_decay': 1e-4,         # 权重衰减
        'epochs': 50,                 # 训练轮数
        'fp16': True if torch.cuda.is_available() else False,  # 混合精度
        'early_stop_patience': 10,    # 早停耐心值
        'ckpt_path': 'egnn_classifier_best.pth'  # 模型保存路径
    }
    
    # 1. 创建数据集
    # full_dataset = GraphClassificationDataset(
    #     num_samples=config['num_samples'],
    #     n_nodes_range=config['n_nodes_range'],
    #     n_feat=config['n_feat'],
    #     x_dim=config['x_dim'],
    #     num_classes=config['num_classes']
    # )
    data_path = Path("get_graph_edge_attr/data/pyg_graph_datas_495_train_edge_attr.pt")
    data_path2 = Path("get_graph_edge_attr/data/pyg_graph_datas_117_train_edge_attr.pt")
    data_train_list = torch.load(data_path,weights_only=False)
    data_test_list = torch.load(data_path2,weights_only=False)


    np.random.seed(42)
    idx = np.random.permutation(len(data_train_list))
    train_idx = idx[:int(0.8 * len(idx))]
    val_idx = idx[int(0.8 * len(idx)):]
    

    train_dataset = [data_train_list[i] for i in train_idx]
    val_dataset = [data_train_list[i] for i in val_idx]
    test_dataset = data_test_list

    # 划分训练/验证/测试集 (8:1:1)
    # train_size = int(0.8 * len(full_dataset))
    # val_size = int(0.1 * len(full_dataset))
    # test_size = len(full_dataset) - train_size - val_size
    # train_dataset, val_dataset, test_dataset = random_split(
    #     full_dataset, [train_size, val_size, test_size],
    #     generator=torch.Generator().manual_seed(42)
    # )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], 
        shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], 
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], 
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    
    # 2. 创建模型
    model = EGNNClassifier(
        in_node_nf=config['n_feat'],
        hidden_nf=config['hidden_nf'],
        num_classes=config['num_classes'],
        in_edge_nf=1,
        n_layers=config['n_layers'],
        device=config['device'],
        attention=config['attention'],
        normalize=config['normalize'],
        tanh=config['tanh'],
        dropout=config['dropout']
    ).to(config['device'])
    
    # 3. 训练模型
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=config
    )
    train_logs, val_logs, test_acc = trainer.fit()
    
    # 4. 特征更新示例（单独提取特征）
    print("\n=== 特征更新示例 ===")
    sample_batch = next(iter(test_loader))
    h = sample_batch['h'].to(config['device'])
    x = sample_batch['x'].to(config['device'])
    edges = sample_batch['edges'].to(config['device'])
    edge_attr = sample_batch['edge_attr'].to(config['device'])
    batch_idx = sample_batch['batch_idx'].to(config['device'])
    
    model.eval()
    with torch.no_grad():
        logits, node_feat, graph_feat = model(h, x, edges, edge_attr, batch_idx)
    
    print(f"原始节点特征形状: {h.shape}")
    print(f"更新后节点特征形状: {node_feat.shape}")
    print(f"图级特征形状: {graph_feat.shape}")
    print(f"分类预测形状: {logits.shape}")
