import numpy as np
import torch
import get_128d_feat
from pathlib import Path

FEAT_DIR = Path("get_features\get_stru\data\stru_feat2")
FEAT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_CACHE = "get_features\\get_stru\\data\\all_graphs2.pt"
RESULT_CACHE = "get_features\\get_stru\\data\\results2.pt"

# 预训练及提取128-d特征
def extract_128d_feature(all_graphs):
    # 划分数据
    pretrain_graphs, rest_graphs = get_128d_feat.split_pretrain_set(all_graphs, 0.3)

    # 预训练
    encoder = get_128d_feat.pretrain_gat(pretrain_graphs, epochs=100)

    # 冻结
    for p in encoder.parameters():
        p.requires_grad = False

    # 给所有蛋白提特征（包括预训练用过的）
    struct_embeddings = []
    for g in all_graphs:
        z = get_128d_feat.extract_struct_embedding(encoder, g)
        struct_embeddings.append(z)
    
    return struct_embeddings


print("✅ 加载缓存数据...")
all_graphs = torch.load(GRAPH_CACHE, map_location="cpu", weights_only=False)
results = torch.load(RESULT_CACHE, weights_only=False)

print(f"✅ 加载完成：{len(all_graphs)} 个蛋白结构图")
struct_features = extract_128d_feature(all_graphs)
for i, (feat, result) in enumerate(zip(struct_features, results)):
    key = f"{result['pdb_id']}_{result['chain']}"
    i+=1
    save_path = FEAT_DIR / f"{i:4d}_{key}_struct.npz"

    np.savez_compressed(
        save_path,
        pdb_id      = result["pdb_id"],
        chain       = result["chain"],
        seq         = result["seq"],
        struct_feat = feat,              # ✅ 128D
        label       = result["label"],
        coords      = result["coords"]
    )
print(f"特征：{struct_features},长度：{len(struct_features)}")
print(f"✅ 已保存 {len(struct_features)} 个蛋白结构特征")