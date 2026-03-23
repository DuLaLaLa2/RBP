from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
import os

STRUCT_DIR = Path(r"get_features\\get_stru\\data\\stru_feat2")
SEQ_DIR    = Path(r"get_features\\get_seq\\data\\seq_feat2")

# 统一解析
def parse_key(filename):
    name = filename.stem
    parts = name.split("_")
    return f"{parts[1]}_{parts[2]}"

def feat_align():
    samples = []
    for sf in STRUCT_DIR.glob("*.npz"):
        sd = np.load(sf)
        struct_feat = sd["struct_feat"]
        label = sd["label"]
        coords = sd["coords"]

        key = sf.stem.replace("_struct", "")
        seq_file = SEQ_DIR / f"{key}.npz"

        if not seq_file.exists():
            print(f"❌ missing seq: {key}")
            continue

        qd = np.load(seq_file)
        seq_feat = qd["seq_feat"]
        samples.append((struct_feat, seq_feat, label, coords))
    
    return samples

# 构建序列边（一阶 + 二阶）
def build_seq_edges(num_nodes: int):
    edges = []
    
    # 一阶序列边 (i, i+1)
    if num_nodes > 1:
        row_1 = torch.arange(num_nodes - 1, dtype=torch.long)
        col_1 = row_1 + 1
        edge_1 = torch.stack([row_1, col_1], dim=0)
        edges.append(edge_1)
        edges.append(edge_1.flip(0))  # 反向
    
    # 二阶序列边 (i, i+2)
    if num_nodes > 2:
        row_2 = torch.arange(num_nodes - 2, dtype=torch.long)
        col_2 = row_2 + 2
        edge_2 = torch.stack([row_2, col_2], dim=0)
        edges.append(edge_2)
        edges.append(edge_2.flip(0))  # 反向
    
    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.cat(edges, dim=1)

# 解析标签
def parse_node_labels(node_labels, num_nodes):
    if isinstance(node_labels, np.ndarray) and node_labels.ndim == 0:
        node_labels = node_labels.item()
    if isinstance(node_labels, np.str_):
        node_labels = str(node_labels)
    
    if not isinstance(node_labels, str):
        raise TypeError(f"标签必须是字符串，当前类型：{type(node_labels)}")

    node_labels = node_labels.strip()
    labels = []
    for c in node_labels:
        if c == "0":
            labels.append(0)
        elif c == "1":
            labels.append(1)
        else:
            raise ValueError(f"非法字符：{c}")

    labels = np.array(labels, dtype=np.int64)
    assert len(labels) == num_nodes, f"标签长度不匹配：{len(labels)} vs {num_nodes}"
    return labels

# ======================= ✅ 核心修改：5维 One-Hot 边特征 =======================
def build_edge_features(edge_index, pos, edge_seq, num_nodes):
    num_edges = edge_index.shape[1]
    # 5 维 One-Hot：[一阶, 二阶, 0~5, 5~10, 10~15]
    edge_feat = torch.zeros((num_edges, 5), dtype=torch.float)

    # 1. 拆分：一阶边集合 / 二阶边集合
    first_order_set = set()
    second_order_set = set()

    # 遍历所有序列边，区分一阶、二阶
    for i in range(edge_seq.shape[1]):
        u = edge_seq[0, i].item()
        v = edge_seq[1, i].item()
        diff = abs(u - v)
        if diff == 1:
            first_order_set.add((u, v))
        elif diff == 2:
            second_order_set.add((u, v))

    # 2. 对每条边赋值 one-hot
    for i in range(num_edges):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()

        if (u, v) in first_order_set:
            edge_feat[i, 0] = 1.0          # 一阶序列边
        elif (u, v) in second_order_set:
            edge_feat[i, 1] = 1.0          # 二阶序列边

    # 3. 计算欧氏距离，赋值空间区间 one-hot
    u_nodes = edge_index[0]
    v_nodes = edge_index[1]
    pos_u = pos[u_nodes]
    pos_v = pos[v_nodes]
    distances = torch.norm(pos_u - pos_v, dim=1)

    edge_feat[distances < 5, 2] = 1.0               # 0~5 Å
    edge_feat[(distances >= 5) & (distances < 10), 3] = 1.0  # 5~10 Å
    edge_feat[(distances >= 10) & (distances <= 15), 4] = 1.0 # 10~15 Å

    return edge_feat

# 构建 PyG Data
def build_pyg_data_list(samples, radius=8.0):
    data_list = []
    for idx, (struct_feat, seq_feat, node_labels, coords) in enumerate(samples):
        struct_feat = torch.tensor(struct_feat, dtype=torch.float)
        seq_feat = torch.tensor(seq_feat, dtype=torch.float)
        num_nodes = struct_feat.shape[0]
        labels = parse_node_labels(node_labels, num_nodes)
        y = torch.tensor(labels, dtype=torch.long)
        pos = torch.tensor(coords, dtype=torch.float)

        edge_seq = build_seq_edges(num_nodes)
        edge_spa = radius_graph(pos, r=radius, loop=False)
        edge_index = torch.cat([edge_seq, edge_spa], dim=1)

        # ✅ 生成 5 维边特征
        edge_attr = build_edge_features(edge_index, pos, edge_seq, num_nodes)

        data = Data(
            struct_feat=struct_feat,
            seq_feat=seq_feat,
            x=torch.cat([struct_feat, seq_feat], dim=-1),
            edge_index=edge_index,
            edge_attr=edge_attr,  # [E, 5]
            pos=pos,
            y=y
        )
        data_list.append(data)
    return data_list

# 保存数据集
def save_pyg_dataset(samples, save_path, radius=8.0):
    data_list = build_pyg_data_list(samples, radius=radius)
    print(f"数据集构建完成：{len(data_list)} 张图")
    print(f"单张图信息：{data_list[0]}")
    os.makedirs(save_path, exist_ok=True)
    save_file = save_path / "pyg_graph_datas_117_train_edge_attr.pt"
    torch.save(data_list, save_file)
    print(f"✅ 数据集已保存至：{save_file}")

if __name__ == "__main__":
    save_path = Path(r"get_graph_edge_attr/data")
    samples = feat_align()
    save_pyg_dataset(samples, save_path)
