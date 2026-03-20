import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
import random
import os

from edge_predictor import EdgePredictor


TRAIN_DATA_PATH = r"get_protain_nodes\\test\data\\train_graphs_for_gan.pt"
OUTPUT_DIR = r"get_protain_nodes\\test\\model"

NODE_FEATURE_DIM = 1408
HIDDEN_DIM = 256
EPOCHS = 50
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
EDGE_SAMPLE_RATIO = 0.2
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def sample_edges_and_non_edges(graph, sample_ratio=0.1):
    """
    从图中采样正样本（有边）和负样本（无边）
    """
    n_nodes = graph.x.size(0)
    edge_index = graph.edge_index
    
    # 构建邻接集合，加速查找
    edge_set = set()
    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        edge_set.add((u, v))
    
    # 正样本：采样部分真实边
    n_edges = edge_index.size(1)
    n_sample = max(1, int(n_edges * sample_ratio))
    
    positive_samples = []
    sampled_edges_idx = np.random.choice(n_edges, min(n_sample, n_edges), replace=False)
    for idx in sampled_edges_idx:
        u, v = edge_index[0, idx].item(), edge_index[1, idx].item()
        positive_samples.append((u, v))
    
    # 负样本：采样无边节点对
    negative_samples = []
    n_neg_sample = len(positive_samples) # * 2  # 负样本多一点
    
    attempts = 0
    max_attempts = n_neg_sample * 10
    while len(negative_samples) < n_neg_sample and attempts < max_attempts:
        u = random.randint(0, n_nodes - 1)
        v = random.randint(0, n_nodes - 1)
        if u != v and (u, v) not in edge_set:
            negative_samples.append((u, v))
        attempts += 1
    
    return positive_samples, negative_samples


def create_edge_dataset(graphs):
    """
    创建边预测数据集
    """
    all_pairs = []
    all_labels = []
    all_graph_indices = []
    
    for graph_idx, graph in enumerate(graphs):
        pos_samples, neg_samples = sample_edges_and_non_edges(graph, EDGE_SAMPLE_RATIO)
        
        for u, v in pos_samples:
            all_pairs.append((u, v))
            all_labels.append(1)
            all_graph_indices.append(graph_idx)
        
        for u, v in neg_samples:
            all_pairs.append((u, v))
            all_labels.append(0)
            all_graph_indices.append(graph_idx)
    
    return all_pairs, torch.tensor(all_labels, dtype=torch.long), torch.tensor(all_graph_indices, dtype=torch.long)


def train():
    set_seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    print(f"加载训练数据: {TRAIN_DATA_PATH}")
    train_data = torch.load(TRAIN_DATA_PATH, weights_only=False, map_location='cpu')
    print(f"训练图数量: {len(train_data)}")
    
    print("创建边预测数据集...")
    pairs, labels, graph_indices = create_edge_dataset(train_data)
    print(f"总样本数: {len(pairs)}, 正样本: {sum(labels)}, 负样本: {len(labels) - sum(labels)}")
    
    # 创建DataLoader
    class EdgeDataset(torch.utils.data.TensorDataset):
        def __init__(self, pairs, labels, graphs, graph_indices):
            self.pairs = pairs
            self.labels = labels
            self.graphs = graphs
            self.graph_indices = graph_indices
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            u, v = self.pairs[idx]
            g = self.graphs[self.graph_indices[idx]]
            node_i = g.x[u].float()
            node_j = g.x[v].float()
            label = self.labels[idx]
            return node_i, node_j, label
    
    dataset = EdgeDataset(pairs, labels, train_data, graph_indices)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = EdgePredictor(NODE_FEATURE_DIM, HIDDEN_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    best_loss = float('inf')
    
    print("\n开始训练边预测器...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        n_batches = 0
        correct = 0
        total = 0
        
        for node_i, node_j, labels_batch in loader:
            node_i = node_i.to(device)
            node_j = node_j.to(device)
            labels_batch = labels_batch.float().to(device)
            
            optimizer.zero_grad()
            logits = model(node_i, node_j)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)
        
        avg_loss = total_loss / n_batches
        acc = correct / total
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'edge_predictor.pt'))
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss {avg_loss:.4f} | Acc {acc:.4f}")
    
    print(f"\n边预测器训练完成! 最佳loss: {best_loss:.4f}")
    print(f"模型保存至: {os.path.join(OUTPUT_DIR, 'edge_predictor.pt')}")


if __name__ == "__main__":
    train()