import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

# =====================================================
# 0. 新增：VGMM 筛选模块（用于判断生成节点是否属于正类）
# =====================================================
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA


class VGMMValidator:
    def __init__(self, n_components=2, threshold=0.3, n_pca=64):
        self.vgmm = BayesianGaussianMixture(
            n_components=n_components,
            covariance_type='full',
            weight_concentration_prior_type='dirichlet_process',
            max_iter=200,
            random_state=42
        )
        self.threshold = threshold
        self.n_pca = n_pca
        self.is_fitted = False
        self.n_components = n_components 

    def fit(self, feats):
        """
        用真实正类节点特征训练VGMM
        feats: [N, feat_dim] numpy array
        """
        if feats.shape[0] < self.n_components:
            print(f"[WARN] 正类节点数不足 ({feats.shape[0]}), 跳过VGMM训练")
            return False
        
        if feats.shape[1] > self.n_pca:
            self.pca = PCA(n_components=min(self.n_pca, feats.shape[1]))
            feats = self.pca.fit_transform(feats)
        
        self.vgmm.fit(feats)
        self.is_fitted = True
        print(f"[VGMM] 训练完成, 使用 {feats.shape[1]} 维特征, n_components={self.n_components}")
        
        # 保存训练时的得分分布用于调试
        self.train_scores = self.vgmm.score_samples(feats)
        print(f"[VGMM] 训练得分分布: min={self.train_scores.min():.2f}, max={self.train_scores.max():.2f}, mean={self.train_scores.mean():.2f}")
        return True

    def validate(self, feats):
        """
        验证合成节点是否属于正类
        feats: [N, feat_dim] numpy array
        返回: (合格节点的mask, 每个节点的得分)
        """
        if not self.is_fitted:
            return np.ones(feats.shape[0], dtype=bool), np.zeros(feats.shape[0])
        
        if hasattr(self, 'pca'):
            feats = self.pca.transform(feats)
        
        scores = self.vgmm.score_samples(feats)
        
        # 调试：打印生成样本的得分分布
        print(f"[VGMM] 生成样本得分: min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}")
        
        # 用相对得分：与训练集得分比较
        # 如果生成样本得分 >= 训练集得分的某个分位数，则认为合格
        train_min = self.train_scores.min()
        train_mean = self.train_scores.mean()
        
        # 合格条件：得分 >= 训练集均值 - 1倍标准差
        train_std = self.train_scores.std()
        threshold_score = train_mean - 0.5 * train_std
        
        mask = scores >= threshold_score
        print(f"[VGMM] 阈值: {threshold_score:.2f}, 合格数: {mask.sum()}/{len(mask)}")
        
        # 归一化到 [0,1] 作为概率
        all_scores = np.concatenate([self.train_scores, scores])
        min_s, max_s = all_scores.min(), all_scores.max()
        probs = (scores - min_s) / (max_s - min_s + 1e-8)
        
        return mask, probs


def train_vgmm(data_list, device, n_components=2, threshold=0.3, n_pca=64):
    """
    从训练数据中提取正类节点，训练VGMM
    """
    pos_feats = []
    
    for data in data_list:
        data = data.to(device)
        pos_idx = (data.y == 1).nonzero(as_tuple=True)[0]
        if len(pos_idx) == 0:
            continue
        
        feats = torch.cat(
            [data.struct_feat[pos_idx], data.seq_feat[pos_idx]],
            dim=-1
        ).cpu().numpy()
        pos_feats.append(feats)
    
    pos_feats = np.vstack(pos_feats)
    print(f"[VGMM] 收集到 {pos_feats.shape[0]} 个正类节点，维度 {pos_feats.shape[1]}")
    
    validator = VGMMValidator(
        n_components=n_components,
        threshold=threshold,
        n_pca=n_pca
    )
    validator.fit(pos_feats)
    
    return validator


# =====================================================
# 1. WGAN-GP：节点特征生成器（只生成特征，不管边）
# =====================================================

class NodeGenerator(nn.Module):
    """
    WGAN-GP Generator
    输入:
        z: 随机噪声 [B, z_dim]
        c: 条件向量（正类原型）[B, cond_dim]
    输出:
        fake_feat: 合成节点特征 [B, feat_dim]
    """
    def __init__(self, z_dim=64, cond_dim=128, feat_dim=1408):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + cond_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, feat_dim)
        )

    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=-1))

class NodeDiscriminator(nn.Module):
    """
    WGAN-GP 判别器
    """
    def __init__(self, feat_dim=1408, cond_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + cond_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x, c):
        return self.net(torch.cat([x, c], dim=-1))

def gradient_penalty(D, real_x, fake_x, c, device):
    alpha = torch.rand(real_x.size(0), 1, device=device)
    alpha = alpha.expand_as(real_x)

    interpolated = alpha * real_x + (1 - alpha) * fake_x
    interpolated.requires_grad_(True)

    d_interpolated = D(interpolated, c)

    grad = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    grad = grad.view(grad.size(0), -1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()

def train_node_gan_wgangp(
    data_list,
    device,
    epochs=100,
    batch_size=64,
    lambda_gp=10.0,
    lambda_anchor=0.0  # ✅ 默认关闭 anchor loss
):
    """
    使用 WGAN-GP 训练节点特征生成器
    【重要】：本版本不再依赖 anchor 节点，而是使用正类原型作为条件
    """

    # =====================================================
    # 1. 收集所有真实正类节点特征
    # =====================================================
    real_feats = []
    cond_feats = []

    for data in data_list:
        data = data.to(device)
        pos_idx = (data.y == 1).nonzero(as_tuple=True)[0]
        if len(pos_idx) == 0:
            continue

        # 拼接 struct + seq 作为真实特征
        real_feats.append(
            torch.cat(
                [data.struct_feat[pos_idx], data.seq_feat[pos_idx]],
                dim=-1
            )
        )

        # 条件：struct_feat（后面会算 prototype）
        cond_feats.append(data.struct_feat[pos_idx])

    real_feats = torch.cat(real_feats, dim=0)
    cond_feats = torch.cat(cond_feats, dim=0)

    print(f"[WGAN] Total positive nodes: {real_feats.size(0)}")

    # =====================================================
    # 2. 计算正类 prototype（全局条件）
    # =====================================================
    c_proto = cond_feats.mean(dim=0, keepdim=True)  # [1, 128]

    # =====================================================
    # 3. 初始化 Generator / Discriminator
    # =====================================================
    G = NodeGenerator(
        z_dim=64,
        cond_dim=c_proto.size(1),
        feat_dim=real_feats.size(1)
    ).to(device)

    D = NodeDiscriminator(
        feat_dim=real_feats.size(1),
        cond_dim=c_proto.size(1)
    ).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

    # =====================================================
    # 4. WGAN-GP 训练循环
    # =====================================================
    n_samples = real_feats.size(0)
    steps_per_epoch = max(1, n_samples // batch_size)

    for epoch in range(epochs):
        perm = torch.randperm(n_samples)

        for step in range(steps_per_epoch):
            idx = perm[step * batch_size:(step + 1) * batch_size]
            real_x = real_feats[idx].to(device)

            # 条件向量：prototype（复制 batch 份）
            c = c_proto.repeat(real_x.size(0), 1).to(device)

            # =========================
            # Train Discriminator
            # =========================
            for _ in range(5):
                z = torch.randn(real_x.size(0), 64, device=device)
                fake_x = G(z, c).detach()

                d_real = D(real_x, c).mean()
                d_fake = D(fake_x, c).mean()

                gp = gradient_penalty(D, real_x, fake_x, c, device)

                loss_D = d_fake - d_real + lambda_gp * gp

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

            # =========================
            # Train Generator
            # =========================
            z = torch.randn(real_x.size(0), 64, device=device)
            fake_x = G(z, c)

            adv_loss = -D(fake_x, c).mean()

            # ✅ 不再使用 anchor loss（保留接口是为了兼容）
            loss_G = adv_loss

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        if epoch % 10 == 0:
            print(
                f"[WGAN] Epoch {epoch:03d} | "
                f"D Loss: {loss_D.item():.4f} | "
                f"G Loss: {loss_G.item():.4f}"
            )

    return G


# =====================================================
# 2. 自监督 Edge Builder（学习如何连边）
# =====================================================

class EdgeBuilder(nn.Module):
    """
    自监督边预测模型（Link Predictor）
    使用 Dropout 以支持 MC Dropout 不确定性估计
    """
    def __init__(self, node_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h_i, h_j):
        """
        输入两个节点特征，输出连边概率（logit）
        """
        h = torch.cat([h_i, h_j], dim=-1)
        return self.mlp(h).squeeze(-1)


# =====================================================
# 3. 训练 Edge Builder（自监督补边）
# =====================================================

def train_edge_builder(data_list, device, epochs=20, lr=1e-3):
    """
    在训练集图上自监督训练 Edge Builder
    """
    model = EdgeBuilder(node_dim=1408).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for data in data_list:
            data = data.to(device)

            if data.x is None:
                data.x = torch.cat(
                    [data.struct_feat, data.seq_feat],
                    dim=-1
                )

            # 正样本：真实边
            pos_edge_index = data.edge_index

            # 负样本：随机不存在的边
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=data.x.size(0),
                num_neg_samples=pos_edge_index.size(1)
            )

            # 构造训练样本
            def edge_loss(edge_index, label):
                i, j = edge_index
                logits = model(data.x[i], data.x[j])
                return F.binary_cross_entropy_with_logits(
                    logits,
                    torch.full_like(logits, label)
                )

            loss = edge_loss(pos_edge_index, 1.0) + edge_loss(neg_edge_index, 0.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[EdgeBuilder] Epoch {epoch:03d} | Loss {total_loss:.4f}")

    return model

# =====================================================
# 4. 置信模块：MC Dropout + 结构先验
# =====================================================

@torch.no_grad()
def confidence_edge_selection(
    edge_model,
    new_feat,
    old_feats,
    deg_old,
    max_deg,
    T=10,
    prob_thresh=0.7,
    var_thresh=0.02,
):
    """
    为一个新节点选择可信边（✅ 工程安全版）
    """
    edge_model.train()  # 开启 Dropout（MC Dropout）

    N = old_feats.size(0)          # ✅ 明确旧节点数
    deg_old = deg_old[:N]          # ✅ 强制对齐

    probs = []

    for _ in range(T):
        logits = edge_model(
            new_feat.repeat(N, 1),
            old_feats
        )
        probs.append(torch.sigmoid(logits))

    probs = torch.stack(probs, dim=0)  # [T, N]

    mean_p = probs.mean(dim=0)
    var_p = probs.var(dim=0)

    # ✅ 显式保证 mask 长度 == N
    mask = (mean_p > prob_thresh) & (var_p < var_thresh)
    mask = mask[:N]

    candidates = mask.nonzero(as_tuple=True)[0]

    selected = []

    for idx in candidates.tolist():
        # ✅【关键 1】索引合法性
        if idx < 0 or idx >= N:
            continue

        # ✅【关键 2】度约束（deg_old 已对齐）
        if deg_old[idx] >= max_deg:
            continue

        selected.append(idx)

    return selected

# =====================================================
# 5. 图增强主函数（新增 VGMM 筛选 + 置信度筛选）
# =====================================================

def augment_graph_with_confidence(
    data,
    G,
    edge_model,
    device,
    target_ratio=1.0,
    z_dim=64,
    vgmm_validator=None,
    max_retry=3,
    vgmm_threshold=0.5
):
    """
    完整增强流程：
    1. WGAN 生成新节点特征
    2. VGMM 筛选（判断是否属于正类）
    3. Edge Builder + 置信模块自动补边
    
    参数:
        vgmm_validator: 训练好的VGMM验证器，如果为None则跳过VGMM筛选
        max_retry: 最大重试次数（当合格节点不足时）
        vgmm_threshold: VGMM判断正类的概率阈值
    """

    # =====================================================
    # 0. 基础防御：构造 data.x
    # =====================================================
    if data.x is None:
        data.x = torch.cat([data.struct_feat, data.seq_feat], dim=-1)

    data = data.to(device)

    # =====================================================
    # 1. 计算需要生成多少正类节点（✅ 比例在这里生效）
    # =====================================================
    pos_idx = (data.y == 1).nonzero(as_tuple=True)[0]
    neg_idx = (data.y == 0).nonzero(as_tuple=True)[0]

    n_pos = pos_idx.numel()
    n_neg = neg_idx.numel()

    if n_pos == 0:
        return data

    target_pos = int(n_neg * target_ratio)
    need = max(0, target_pos - n_pos)

    if need == 0:
        return data

    # =====================================================
    # 2. 计算 prototype 条件向量（与 WGAN 训练一致）
    # =====================================================
    c_proto = data.struct_feat[pos_idx].mean(dim=0, keepdim=True)  # [1, 128]

    # =====================================================
    # 3. 生成节点 + VGMM 筛选（支持重试）
    # =====================================================
    old_num_nodes = data.x.size(0)
    valid_new_feats = []
    valid_scores = []

    for retry in range(max_retry):
        if len(valid_new_feats) >= need:
            break
        
        batch_need = need - len(valid_new_feats)
        
        # 生成候选节点
        z = torch.randn(batch_need, z_dim, device=device)
        c = c_proto.repeat(batch_need, 1).to(device)  # 扩展到batch维度
        fake_feats = G(z, c)  # [batch_need, feat_dim]
        
        if vgmm_validator is not None and vgmm_validator.is_fitted:
            # VGMM 筛选
            fake_np = fake_feats.cpu().numpy()
            mask, scores = vgmm_validator.validate(fake_np)
            
            valid_idx = mask.nonzero()[0]
            for idx in valid_idx:
                valid_new_feats.append(fake_feats[idx])
                valid_scores.append(scores[idx])
            
            print(f"[VGMM] Retry {retry+1}: 生成了 {batch_need} 个, 合格 {len(valid_idx)} 个")
        else:
            # 无VGMM时全部接受
            valid_new_feats.extend([fake_feats[i] for i in range(batch_need)])
            valid_scores.extend([1.0] * batch_need)

    valid_new_feats = torch.stack(valid_new_feats[:need], dim=0)
    print(f"[VGMM] 最终合格节点: {len(valid_new_feats)}/{need}")

    # =====================================================
    # 4. 把新节点"注册"进图（✅ 关键顺序）
    # =====================================================
    data.x = torch.cat([data.x, valid_new_feats], dim=0)

    data.struct_feat = torch.cat(
        [data.struct_feat, valid_new_feats[:, :data.struct_feat.size(1)]],
        dim=0
    )
    data.seq_feat = torch.cat(
        [data.seq_feat, valid_new_feats[:, data.struct_feat.size(1):]],
        dim=0
    )

    data.y = torch.cat(
        [data.y,
         torch.ones(len(valid_new_feats), device=device, dtype=data.y.dtype)],
        dim=0
    )
    # ✅ 此时 data.num_nodes 已经是正确的

    # =====================================================
    # 5. 置信度补边（只连向旧节点，防止新节点互连）
    # =====================================================
    deg = torch.bincount(
        data.edge_index[0],
        minlength=old_num_nodes
    )

    # 正类平均度作为上限先验
    max_deg = int(
        deg[pos_idx].float().mean() + 2 * deg[pos_idx].float().std()
    )

    new_edges = []

    for i in range(len(valid_new_feats)):
        new_id = old_num_nodes + i

        neighbors = confidence_edge_selection(
            edge_model=edge_model,
            new_feat=data.x[new_id],
            old_feats=data.x[:old_num_nodes],
            deg_old=deg,
            max_deg=max_deg
        )

        for j in neighbors:
            j = int(j)

            # ✅【关键防御】只允许连向旧节点
            if j < 0 or j >= old_num_nodes:
                continue
            if j >= old_num_nodes:
                print(f"[WARN] illegal neighbor j={j}, old_num_nodes={old_num_nodes}")
            new_edges.append([new_id, j])
            new_edges.append([j, new_id])
            
    # =====================================================
    # 6. 拼接 edge_index
    # =====================================================
    if len(new_edges) > 0:
        edge_add = torch.tensor(
            new_edges,
            device=device,
            dtype=torch.long
        ).t()

        data.edge_index = torch.cat(
            [data.edge_index, edge_add],
            dim=1
        )

    # ✅ 最终安全检查（调试阶段建议保留）
    # ✅ 用可信的 x.size(0) 替代 PyG 推断的 num_nodes
    assert data.edge_index.max().item() < data.x.size(0), \
    f"edge_index out of range! max_idx={data.edge_index.max().item()}, real_num_nodes={data.x.size(0)}"

    return data

if __name__ == "__main__":

    import torch
    from copy import deepcopy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =====================================================
    # Stage 0: 加载数据 + 划分数据集
    # =====================================================
    data_list = torch.load(
        "D:/dev/Coding/RBP_v2/get_graph_data/data/pyg_graph_datas_495_train.pt",
        weights_only=False
    )

    n_total = len(data_list)
    n_train = int(0.7 * n_total)
    n_val   = int(0.15 * n_total)

    train_data = data_list[:n_train]
    val_data   = data_list[n_train:n_train + n_val]
    test_data  = data_list[n_train + n_val:]

    print(f"[INFO] Train / Val / Test = "
          f"{len(train_data)} / {len(val_data)} / {len(test_data)}")


    # =====================================================

    # =====================================================
    # Stage 0.5: 训练 VGMM（新增）
    # =====================================================
    print("\n[Stage 0.5] Training VGMM for node validation...")
    
    vgmm_validator = train_vgmm(
        data_list=train_data,
        device=device,
        n_components=2,
        threshold=0.3,
        n_pca=64
    )

    # =====================================================
    # Stage 1: 自监督训练 Edge Builder（补边模块）
    # =====================================================
    print("\n[Stage 1] Training Edge Builder (self-supervised)...")

    edge_builder = train_edge_builder(
        data_list=train_data,
        device=device,
        epochs=50,
        lr=1e-3
    )

    # 冻结 Edge Builder（只用于推理）
    edge_builder.eval()
    for p in edge_builder.parameters():
        p.requires_grad = False

    # =====================================================
    # Stage 2: 训练 WGAN-GP（节点特征生成器）
    # =====================================================
    print("\n[Stage 2] Training WGAN-GP for node feature generation...")

    G = train_node_gan_wgangp(
        data_list=train_data,
        device=device,
        epochs=100,
        batch_size=64,
        lambda_gp=10.0,
        lambda_anchor=0.0  # ✅ 不再使用 anchor loss
    )

    G.eval()
    for p in G.parameters():
        p.requires_grad = False

    # =====================================================
    # Stage 3: 图增强（WGAN + VGMM筛选 + Edge Builder + 置信模块）
    # =====================================================
    print("\n[Stage 3] Augmenting training graphs with VGMM + confidence-aware edges...")

    enhanced_train = []

    for idx, data in enumerate(train_data):
        data_aug = augment_graph_with_confidence(
            data=deepcopy(data),   # ✅ 防止原图被原地修改
            G=G,
            edge_model=edge_builder,
            device=device,
            target_ratio=0.4,       # 正负样本 2:5
            vgmm_validator=vgmm_validator,
            max_retry=3,
            vgmm_threshold=0.5
        )
        enhanced_train.append(data_aug)
        
        if idx % 50 == 0:
            print(f"  Augmented {idx}/{len(train_data)} graphs")
            print(f"data_aug[{idx}]:{data_aug}")
    # =====================================================
    # Stage 4: 拼回 val / test（不增强）
    # =====================================================
    enhanced_all = enhanced_train + val_data + test_data

    print(f"\n[INFO] Final dataset size = {len(enhanced_all)}")

    torch.save(
        enhanced_all,
        "D:/dev/Coding/RBP_v2/get_pre_result/data/pyg_495—2_enhance_datas.pt"
    )

    print("[DONE] Graph augmentation pipeline finished successfully.")
