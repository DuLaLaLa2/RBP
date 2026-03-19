import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

def compute_graph_stats(data_list, name):
    """计算图的统计特征"""
    stats = {
        'n_nodes': [],
        'n_edges': [],
        'class_ratio': [],
        'feature_mean': [],
        'feature_std': [],
    }
    
    for d in data_list:
        stats['n_nodes'].append(d.x.size(0))
        stats['n_edges'].append(d.edge_index.size(1) if d.edge_index is not None else 0)
        
        labels = d.y.cpu().numpy() if isinstance(d.y, torch.Tensor) else d.y
        if hasattr(labels, 'tolist'):
            labels = labels.tolist()
        pos = sum(1 for l in labels if l == 1)
        neg = len(labels) - pos
        stats['class_ratio'].append(pos / neg if neg > 0 else 0)
        
        stats['feature_mean'].append(d.x.mean().item())
        stats['feature_std'].append(d.x.std().item())
    
    print(f"\n{'='*50}")
    print(f"{name} 统计")
    print(f"{'='*50}")
    print(f"节点数: 均值={np.mean(stats['n_nodes']):.1f}, 标准差={np.std(stats['n_nodes']):.1f}")
    print(f"边数: 均值={np.mean(stats['n_edges']):.1f}, 标准差={np.std(stats['n_edges']):.1f}")
    print(f"类别比例(正/负): 均值={np.mean(stats['class_ratio']):.3f}")
    print(f"特征均值: 均值={np.mean(stats['feature_mean']):.4f}")
    print(f"特征标准差: 均值={np.mean(stats['feature_std']):.4f}")
    
    return stats

def compute_distribution_difference(stats1, stats2, name1, name2):
    """计算两个分布之间的差异"""
    print(f"\n{name1} vs {name2} 分布差异:")
    for key in ['n_nodes', 'n_edges', 'class_ratio']:
        mean1, std1 = np.mean(stats1[key]), np.std(stats1[key])
        mean2, std2 = np.mean(stats2[key]), np.std(stats2[key])
        diff = abs(mean1 - mean2)
        print(f"  {key}: 均值差异={diff:.2f} ({diff/mean1*100:.1f}%)")

# 加载数据
print("加载数据...")
train_data = torch.load("get_protain_nodes/test/data/enhanced_train_graphs.pt", weights_only=False, map_location='cpu')
val_data = torch.load("get_protain_nodes/test/data/val_graphs.pt", weights_only=False, map_location='cpu')
test_data = torch.load("get_graph_data/data/pyg_graph_datas_117_test.pt", weights_only=False, map_location='cpu')

print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")

# 计算统计
train_stats = compute_graph_stats(train_data, "训练集(增强)")
val_stats = compute_graph_stats(val_data, "验证集(原始)")
test_stats = compute_graph_stats(test_data, "测试集(原始)")

# 比较分布差异
print("\n" + "="*50)
print("分布差异分析")
print("="*50)
compute_distribution_difference(train_stats, val_stats, "训练集", "验证集")
compute_distribution_difference(train_stats, test_stats, "训练集", "测试集")
compute_distribution_difference(val_stats, test_stats, "验证集", "测试集")

# 特征空间分析 - 采样并PCA可视化
print("\n" + "="*50)
print("特征空间分析")
print("="*50)

# 采样节点特征
def sample_node_features(data_list, n_samples=100):
    features = []
    for d in data_list[:20]:  # 取前20个图
        n = min(d.x.size(0), n_samples // 20)
        indices = np.random.choice(d.x.size(0), n, replace=False)
        features.append(d.x[indices].numpy())
    return np.vstack(features)

np.random.seed(42)
train_feat = sample_node_features(train_data)
val_feat = sample_node_features(val_data)
test_feat = sample_node_features(test_data)

print(f"采样特征: 训练集={train_feat.shape}, 验证集={val_feat.shape}, 测试集={test_feat.shape}")

# 合并所有特征进行PCA
all_feat = np.vstack([train_feat, val_feat, test_feat])
pca = PCA(n_components=2)
all_pca = pca.fit_transform(all_feat)

train_pca = all_pca[:len(train_feat)]
val_pca = all_pca[len(train_feat):len(train_feat)+len(val_feat)]
test_pca = all_pca[len(train_feat)+len(val_feat):]

print(f"PCA解释方差比: {pca.explained_variance_ratio_}")

# 保存PCA结果
plt.figure(figsize=(10, 8))
plt.scatter(train_pca[:, 0], train_pca[:, 1], alpha=0.5, label='Train', s=10)
plt.scatter(val_pca[:, 0], val_pca[:, 1], alpha=0.5, label='Val', s=10)
plt.scatter(test_pca[:, 0], test_pca[:, 1], alpha=0.5, label='Test', s=10)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Node Features')
plt.legend()
plt.savefig('get_protain_nodes/test/data/distribution_analysis.png', dpi=150)
print("\nPCA图已保存: get_protain_nodes/test/data/distribution_analysis.png")