from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
import os


STRUCT_DIR = Path("get_features\get_stru\data\stru_feat_117_42")
SEQ_DIR    = Path("get_features\get_seq\data\seq_feat2")

# 统一解析
def parse_key(filename):
    # e.g. 1_3pla_L_struct.npz → 3pla_L
    name = filename.stem
    parts = name.split("_")
    return f"{parts[1]}_{parts[2]}"

def feat_align():

    samples = []
    for sf in STRUCT_DIR.glob("*.npz"):
        sd = np.load(sf)

        # print("STRUCT:", sf, sd.files)

        struct_feat = sd["struct_feat"]
        label = sd["label"]
        coords = sd["coords"]

        key = sf.stem.replace("_struct", "")
        seq_file = SEQ_DIR / f"{key}.npz"

        if not seq_file.exists():
            print(f"❌ missing seq: {key}")
            continue

        qd = np.load(seq_file)
        # print("SEQ:", seq_file, qd.files)

        seq_feat = qd["seq_feat"]

        samples.append((struct_feat, seq_feat, label, coords))
    
    # print(f"{samples}\n{len(samples)}") 
    return samples

# 构建序列边
def build_seq_edges(num_nodes: int):
    """
    构建 i <-> i+1 的双向序列边
    """
    row = torch.arange(num_nodes - 1, dtype=torch.long)
    col = row + 1
    edge = torch.stack([row, col], dim=0)
    return torch.cat([edge, edge.flip(0)], dim=1)

# 解析标签
def parse_node_labels(node_labels, num_nodes):
    """
    node_labels 可能是：
      - str
      - np.str_
      - 0-d np.ndarray (scalar string)
    最终返回：
      np.ndarray shape [num_nodes], dtype int64
    """

    # ✅ 情况 1：0-d numpy array（最关键）
    if isinstance(node_labels, np.ndarray) and node_labels.ndim == 0:
        node_labels = node_labels.item()  # 解包成 Python str

    # ✅ 情况 2：numpy string scalar
    if isinstance(node_labels, np.str_):
        node_labels = str(node_labels)

    # ✅ 现在一定是 Python 字符串
    if isinstance(node_labels, str):
        labels = np.fromiter(node_labels, dtype=np.int64)

    else:
        raise TypeError(f"Unsupported label type: {type(node_labels)}")

    assert len(labels) == num_nodes, \
        f"❌ label length {len(labels)} != num_nodes {num_nodes}"

    return labels

# 根据坐标构图 samples → PyG Data 列表
def build_pyg_data_list(samples, radius=14.0):
    data_list = []

    for idx, (struct_feat, seq_feat, node_labels, coords) in enumerate(samples):
        struct_feat = torch.tensor(struct_feat, dtype=torch.float)
        seq_feat    = torch.tensor(seq_feat, dtype=torch.float)
        num_nodes = struct_feat.shape[0]
        labels      = parse_node_labels(node_labels, num_nodes)
        y           = torch.tensor(labels, dtype=torch.long)
        pos         = torch.tensor(coords, dtype=torch.float)

        # num_nodes = struct_feat.size(0)

        # 序列边
        edge_seq = build_seq_edges(num_nodes)

        # 空间邻接边
        edge_spa = radius_graph(pos, r=radius, loop=False)

        # 合并边
        edge_index = torch.cat([edge_seq, edge_spa], dim=1)


        data = Data(
            struct_feat=struct_feat,   # [N, 42]
            seq_feat=seq_feat,         # [N, 1280]
            x=torch.cat([struct_feat, seq_feat], dim=-1),                       # [N, 1408]
            edge_index=edge_index,     # [2, E]
            pos=pos,                   # [N, 3]
            y=y                         # [N] 节点级标签
        )

        data_list.append(data)

    return data_list

# 保存文件
def save_pyg_dataset(samples, save_path, radius=14.0):
    data_list = build_pyg_data_list(samples, radius=radius)
    print(f"data_list info:{data_list[0]},{len(data_list)}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    save_path = save_path / "pyg_graph_datas_117_test_1280_42.pt"
    torch.save(data_list, save_path)


if __name__== "__main__":
    save_path = Path("get_graph_data\data")
    samples = feat_align()
    save_pyg_dataset(samples, save_path)