import numpy as np
import torch
from pathlib import Path

FEAT_DIR = Path("get_features\get_stru\data\stru_feat_117_42")
FEAT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_CACHE = "get_features\\get_stru\\data\\all_graphs2.pt"
RESULT_CACHE = "get_features\\get_stru\\data\\results2.pt"


print("✅ 加载缓存数据...")
all_graphs = torch.load(GRAPH_CACHE, map_location="cpu", weights_only=False)
results = torch.load(RESULT_CACHE, weights_only=False)
print(f"✅ 加载完成：{len(all_graphs)} 个蛋白结构图")

for i, (graph, result) in enumerate(zip(all_graphs, results)):
    key = f"{result['pdb_id']}_{result['chain']}"
    i+=1
    save_path = FEAT_DIR / f"{i}_{key}_struct.npz"

    np.savez_compressed(
        save_path,
        pdb_id      = result["pdb_id"],
        chain       = result["chain"],
        seq         = result["seq"],
        struct_feat = graph["x"].numpy(),              # ✅ 42D
        label       = result["label"],
        coords      = result["coords"]
    )
print(f"✅ 已保存 {len(all_graphs)} 个蛋白结构特征")