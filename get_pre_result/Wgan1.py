import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import random
from tqdm import tqdm
import numpy as np
import os

# =========================
# 【核心修改】Windows绝对路径配置 ↓↓↓
# 替换成你自己的实际文件夹路径（必须改！）
ROOT_PATH = r"D:\\dev\\Coding\\RBP_v2"  # 示例：C:/Users/LiMing/Code/ProteinGraph
DATA_PATH = os.path.join(ROOT_PATH, "get_graph_data\data\pyg_graph_datas_495_train.pt")  # 原始数据路径
SAVE_PATH = os.path.join(ROOT_PATH, "get_pre_result/Wgan1_pyg_enhance_graph_datas.pt")  # 增强数据保存路径
# =========================

# 常量定义
NOISE_DIM = 64
COND_DIM = 128
STRUCT_FEAT_DIM = 128
SEQ_FEAT_DIM = 320
NODE_FEAT_DIM = STRUCT_FEAT_DIM + SEQ_FEAT_DIM
BATCH_SIZE = 64
EPOCHS = 20
LAMBDA_GP = 10.0
LAMBDA_ANCHOR = 0.1
N_CRITIC = 1
SEED = 42
K_NEIGHBORS = 5
DISTANCE_THRESHOLD = 1.0

# 固定随机种子
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# =========================
# Generator 生成器
# =========================
class NodeGenerator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM, cond_dim=COND_DIM, out_dim=NODE_FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, out_dim)
        )

    def forward(self, z, c):
        # z:噪声[B,64], c:条件特征[B,128]
        return self.net(torch.cat([z, c], dim=-1))  # 输出[B,448]

# =========================
# Discriminator 判别器（无需改）
# =========================
class NodeDiscriminator(nn.Module):
    def __init__(self, cond_dim=COND_DIM, in_dim=NODE_FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x, c):
        # x:节点特征[B,448]（真实/生成）, c:条件特征[B,128]
        return self.net(torch.cat([x, c], dim=-1)) # 输出判别得分[B,1]

# =========================
# 梯度惩罚
# =========================
def gradient_penalty(D, real_x, fake_x, c, device):
    batch_size = real_x.size(0)
    # 1. 生成随机插值权重α
    alpha = torch.rand(batch_size, 1, device=device, requires_grad=False).expand_as(real_x)
    # 2. 构建真实/生成节点的插值样本
    interpolated = alpha * real_x + (1 - alpha) * fake_x
    interpolated.requires_grad_(True)
    # 3. 判别器对插值样本的评分
    d_interpolated = D(interpolated, c)
    # 4. 计算插值样本的梯度
    grad = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    # 5. 计算梯度惩罚项：(||grad||_2 - 1)^2的均值
    grad = grad.view(batch_size, -1)
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# =========================
# 训练GAN
# =========================
def train_node_gan_wgangp(
        data_list,
        device,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lambda_gp=LAMBDA_GP,
        lambda_anchor=LAMBDA_ANCHOR,
        label=1
):
    # 筛选指定标签的节点，收集特征和条件
    feats, conds = [], []
    for data in data_list:
        target_idx = (data.y == label).nonzero(as_tuple=True)[0]
        if len(target_idx) == 0:
            continue
        # 拼接结构+序列特征作为真实节点特征
        node_feat = torch.cat([data.struct_feat[target_idx], data.seq_feat[target_idx]], dim=-1)
        feats.append(node_feat.to(device))
        # 条件特征用结构特征
        conds.append(data.struct_feat[target_idx].to(device))

    # 检查是否有指定类别的节点
    if len(feats) == 0:
        raise ValueError(f"数据集中没有{label}类节点，无法训练GAN！")

    real_feats = torch.cat(feats)  # 所有目标类别节点特征集合
    cond_feats = torch.cat(conds)  # 对应的条件特征集合
    num_samples = real_feats.size(0)
    print(f"训练GAN（{label}类）：共收集 {num_samples} 个节点")
    # 初始化模型和优化器
    G = NodeGenerator(out_dim=NODE_FEAT_DIM).to(device)
    D = NodeDiscriminator(in_dim=NODE_FEAT_DIM).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))

    # 训练循环
    for epoch in tqdm(range(epochs), desc="Training GAN"):
        # 随机打乱数据
        perm = torch.randperm(num_samples, device=device)
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        batch_count = 0

        for i in range(0, num_samples, batch_size):
            idx = perm[i:i + batch_size]
            real_x = real_feats[idx]
            c = cond_feats[idx]
            batch_size_current = real_x.size(0)
            if batch_size_current < 2:  # 避免批次太小导致梯度计算错误
                continue
            batch_count += 1

            # 训练判别器
            d_loss_batch = 0.0
            for _ in range(N_CRITIC):
                z = torch.randn(batch_size_current, NOISE_DIM, device=device)
                fake_x = G(z, c).detach()  # 冻结生成器
                d_real = D(real_x, c).mean()  # 真实节点评分
                d_fake = D(fake_x, c).mean()  # 生成节点评分
                gp = gradient_penalty(D, real_x, fake_x, c, device)  # 梯度惩罚
                loss_D = d_fake - d_real + lambda_gp * gp  # WGAN-GP判别器损失
                # 反向传播
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()
                d_loss_batch += loss_D.item()
            epoch_d_loss += d_loss_batch / N_CRITIC

            # 训练生成器
            z = torch.randn(batch_size_current, NOISE_DIM, device=device)
            fake_x = G(z, c)
            adv_loss = -D(fake_x, c).mean()  # 对抗损失：让生成节点评分尽可能高
            anchor_loss = F.mse_loss(fake_x, real_x)  # 锚点损失：让生成特征贴近真实
            loss_G = adv_loss + lambda_anchor * anchor_loss  # 生成器总损失
            # 反向传播
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            epoch_g_loss += loss_G.item()

        # 每10轮打印日志
        if epoch % 10 == 0 and batch_count > 0:
            avg_d_loss = epoch_d_loss / batch_count
            avg_g_loss = epoch_g_loss / batch_count
            print(
                f"Epoch {epoch:03d} | "
                f"D_loss={avg_d_loss:.4f} | "
                f"G_loss={avg_g_loss:.4f} | "
                f"G_adv={adv_loss.item():.4f} | "
                f"anchor={anchor_loss.item():.4f}"
            )
    return G

# =========================
# 距离矩阵计算
# =========================
def calculate_distance_matrix(feat1, feat2, pos1=None, pos2=None, feat_weight=0.7, pos_weight=0.3):
    # 1. 特征余弦距离：1 - 余弦相似度（0=最相似，1=最不相似）
    feat1_norm = F.normalize(feat1, p=2, dim=-1)
    feat2_norm = F.normalize(feat2, p=2, dim=-1)
    cos_sim = torch.mm(feat1_norm, feat2_norm.t())
    feat_dist = 1 - cos_sim
    # 2. 位置欧式距离：归一化到[0,1]
    if pos1 is not None and pos2 is not None:
        pos_dist = torch.cdist(pos1, pos2, p=2)
        pos_dist = pos_dist / pos_dist.max() if pos_dist.max() != 0 else pos_dist
    else:
        pos_dist = torch.zeros_like(feat_dist)
    # 3. 混合距离：加权融合特征和位置距离
    total_dist = feat_weight * feat_dist + pos_weight * pos_dist
    return total_dist

# =========================
# 图增强
# =========================
def augment_graph(data, G_pos, G_neg, device, target_ratio=3.0, k_neighbors=K_NEIGHBORS, dist_threshold=DISTANCE_THRESHOLD):
    data = data.cpu()
    # 筛选正负节点索引
    pos_idx = (data.y == 1).nonzero(as_tuple=True)[0]
    neg_idx = (data.y == 0).nonzero(as_tuple=True)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    # 目标：正负比 1:3
    target_neg = int(n_pos * target_ratio)
    need_neg = max(0, target_neg - n_neg)
    
    # 无需增强，直接返回
    if n_pos == 0 or n_neg == 0:
        return data

    if need_neg == 0:
        print(f"图已满足1:3比例（正类：{n_pos}，负类：{n_neg}），无需增强")
        return data

    print(f"增强图：需要生成 {need_neg} 个负类节点（当前正类：{n_pos}，负类：{n_neg}，目标负类：{target_neg}）")

    # 批量生成负类节点
    G_neg.eval()
    with torch.no_grad():
        # 随机选择锚点节点（用于条件特征，使用负类节点）
        anchor_indices_neg = random.choices(neg_idx.numpy(), k=need_neg)
        c_neg = data.struct_feat[anchor_indices_neg].to(device)
        z_neg = torch.randn(need_neg, NOISE_DIM, device=device)
        fake_feats_neg = G_neg(z_neg, c_neg).cpu()
        fake_struct_neg = fake_feats_neg[:, :STRUCT_FEAT_DIM]
        fake_seq_neg = fake_feats_neg[:, STRUCT_FEAT_DIM:]
        anchor_pos_neg = data.pos[anchor_indices_neg]
        jitter_neg = 0.1 * torch.randn_like(anchor_pos_neg)
        new_pos_neg = anchor_pos_neg + jitter_neg

    # 计算距离矩阵生成边（为新生成的负类节点）
    original_feat = torch.cat([data.struct_feat, data.seq_feat], dim=-1)
    original_pos = data.pos
    dist_matrix = calculate_distance_matrix(fake_feats_neg, original_feat, new_pos_neg, original_pos)
    new_edges = []
    base_N = data.y.size(0)

    for i in range(need_neg):
        new_id = base_N + i
        dist = dist_matrix[i]
        # 筛选距离小于阈值的有效节点
        valid_indices = (dist < dist_threshold).nonzero(as_tuple=True)[0]
        if len(valid_indices) == 0:
            _, topk_indices = torch.topk(-dist, k=min(k_neighbors, len(dist)))
        else:
            valid_dist = dist[valid_indices]
            _, topk_idx_in_valid = torch.topk(-valid_dist, k=min(k_neighbors, len(valid_dist)))
            topk_indices = valid_indices[topk_idx_in_valid]
        for neighbor_id in topk_indices:
            new_edges.append([new_id, neighbor_id.item()])
            new_edges.append([neighbor_id.item(), new_id])

    # 重构图数据
    data.struct_feat = torch.cat([data.struct_feat, fake_struct_neg], dim=0)
    data.seq_feat = torch.cat([data.seq_feat, fake_seq_neg], dim=0)
    data.pos = torch.cat([data.pos, new_pos_neg], dim=0)
    data.y = torch.cat([data.y, torch.zeros(need_neg, dtype=data.y.dtype)], dim=0)
    
    if new_edges:
        edge_add = torch.tensor(new_edges, dtype=torch.long).t()
        all_edges = torch.cat([data.edge_index, edge_add], dim=1)
        all_edges_np = all_edges.numpy().T
        all_edges_np = np.unique(all_edges_np, axis=0)
        data.edge_index = torch.tensor(all_edges_np.T, dtype=torch.long)

    return data

# =========================
# 生成测试数据
# =========================
def generate_test_data(num_graphs=10, min_nodes=50, max_nodes=200):
    # 生成测试数据（用于调试）
    print(f"\n=== 生成测试数据 ===")
    print(f"生成 {num_graphs} 个测试图，每个图节点数范围：{min_nodes}-{max_nodes}")
    test_data = []
    # 随机生成测试图数据
    for i in range(num_graphs):
        num_nodes = random.randint(min_nodes, max_nodes)
        struct_feat = torch.randn(num_nodes, STRUCT_FEAT_DIM)
        seq_feat = torch.randn(num_nodes, SEQ_FEAT_DIM)
        pos = torch.randn(num_nodes, 3)
        y = torch.randint(0, 2, (num_nodes,))
        # 随机生成边（无向）
        edges = []
        for j in range(num_nodes):
            neighbors = random.sample(range(num_nodes), random.randint(1, 5))
            for n in neighbors:
                edges.append([j, n])
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        test_data.append(Data(
            struct_feat=struct_feat, seq_feat=seq_feat, pos=pos, y=y, edge_index=edge_index
        ))
    print(f"测试数据生成完成：共 {len(test_data)} 个图")
    return test_data

# =========================
# 主函数
# =========================
if __name__ == "__main__":
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 初始化 ===")
    print(f"使用设备：{device}")
    print(f"原始数据绝对路径：{DATA_PATH}")
    print(f"增强数据保存绝对路径：{SAVE_PATH}")

    # 加载数据
    data_list = None
    if os.path.exists(DATA_PATH):
        try:
            data_list = torch.load(DATA_PATH)
            print(f"\n=== 加载数据成功 ===")
            print(f"共加载 {len(data_list)} 个图数据")
        except Exception as e:
            print(f"\n[!] 数据文件存在但加载失败：{e}")
            data_list = generate_test_data()
    else:
        print(f"\n[!] 未找到原始数据文件，自动生成测试数据用于演示")
        data_list = generate_test_data()

    # 划分数据集
    n_total = len(data_list)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    train_data = data_list[:n_train]
    val_data = data_list[n_train:n_train + n_val]
    test_data = data_list[n_train + n_val:]
    print(f"\n=== 数据集划分 ===")
    print(f"总数量：{n_total} | 训练集：{len(train_data)} | 验证集：{len(val_data)} | 测试集：{len(test_data)}")
    print(f"训练集第一个图信息：节点数 {train_data[0].num_nodes}，边数 {train_data[0].num_edges}")

    # 训练GAN（分别训练正类和负类生成器）
    print(f"\n=== 开始训练GAN（正类） ===")
    G_pos = train_node_gan_wgangp(train_data, device, label=1)
    print(f"\n=== 开始训练GAN（负类） ===")
    G_neg = train_node_gan_wgangp(train_data, device, label=0)

    # 增强训练集
    print(f"\n=== 开始图增强 ===")
    enhanced_train = []
    for data in tqdm(train_data, desc="Augmenting train data"):
        enhanced_train.append(augment_graph(data, G_pos, G_neg, device))

    # 合并并保存
    enhanced_all = enhanced_train + val_data + test_data
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)  # 自动创建文件夹
    torch.save(enhanced_all, SAVE_PATH)
    print(f"\n=== 运行完成 ===")
    print(f"增强后的数据已保存到绝对路径：{SAVE_PATH}")
    print(f"增强后训练集第一个图信息：节点数 {enhanced_train[0].num_nodes}，边数 {enhanced_train[0].num_edges}")