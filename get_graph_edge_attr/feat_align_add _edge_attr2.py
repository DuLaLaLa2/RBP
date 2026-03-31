from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
import os


STRUCT_DIR = Path("get_features\get_stru\data\stru_feat_495_42")
SEQ_DIR    = Path("get_features\get_seq\data\seq_feat")

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

# 构建序列边（包含一阶和二阶）
def build_seq_edges(num_nodes: int):
    """
    构建序列边：
    - 一阶序列边：i <-> i+1
    - 二阶序列边：i <-> i+2
    返回所有序列边的双向边
    """
    edges = []
    
    # 一阶序列边 (i, i+1)
    row_1 = torch.arange(num_nodes - 1, dtype=torch.long)
    col_1 = row_1 + 1
    edge_1 = torch.stack([row_1, col_1], dim=0)
    edges.append(edge_1)
    edges.append(edge_1.flip(0))  # 双向
    
    # 二阶序列边 (i, i+2)
    if num_nodes > 2:
        row_2 = torch.arange(num_nodes - 2, dtype=torch.long)
        col_2 = row_2 + 2
        edge_2 = torch.stack([row_2, col_2], dim=0)
        edges.append(edge_2)
        edges.append(edge_2.flip(0))  # 双向
    
    return torch.cat(edges, dim=1)

# 解析标签
def parse_node_labels(node_labels, num_nodes):
    """
    安全解析标签：
    - 输入必须是只由 0 和 1 组成的字符串（如 "00110101"）
    - 输出一定是 0 或 1 的 numpy 数组
    """
    # 1. 安全解开各种 numpy 字符串格式
    if isinstance(node_labels, np.ndarray) and node_labels.ndim == 0:
        node_labels = node_labels.item()
    if isinstance(node_labels, np.str_):
        node_labels = str(node_labels)
    
    # 2. 确保是字符串
    if not isinstance(node_labels, str):
        raise TypeError(f"标签必须是字符串，当前类型：{type(node_labels)}")

    # 3. 去除空白（最关键！防止空格/换行导致解析错误）
    node_labels = node_labels.strip()

    # 4. 逐个字符转 0/1
    labels = []
    for c in node_labels:
        if c == "0":
            labels.append(0)
        elif c == "1":
            labels.append(1)
        else:
            # 发现非法字符直接报错，避免训练出问题
            raise ValueError(f"标签只能是0或1，发现非法字符：{c}")

    labels = np.array(labels, dtype=np.int64)

    # 5. 长度校验
    assert len(labels) == num_nodes, \
        f"标签长度不匹配：标签={len(labels)}，残基数={num_nodes}"

    return labels

# 生成边特征（修改版：三种边类型 + 连续距离归一化）
def build_edge_features(edge_index, pos, edge_seq_first, edge_seq_second):
    """
    生成边特征：
    - 边类型 one-hot (3维)：一阶序列边 / 二阶序列边 / 空间边
    - 连续距离 (1维)：归一化到 [0, 1]，使用 14Å 作为最大值
    总维度：4维
    """
    num_edges = edge_index.shape[1]
    edge_feat = torch.zeros((num_edges, 4), dtype=torch.float)
    
    # 构建快速查询集合
    first_order_set = set()
    for i in range(edge_seq_first.shape[1]):
        u, v = edge_seq_first[0, i].item(), edge_seq_first[1, i].item()
        first_order_set.add((u, v))
    
    second_order_set = set()
    for i in range(edge_seq_second.shape[1]):
        u, v = edge_seq_second[0, i].item(), edge_seq_second[1, i].item()
        second_order_set.add((u, v))
    
    # 1. 标记边类型（one-hot）
    for i in range(num_edges):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        
        if (u, v) in first_order_set:
            edge_feat[i, 0] = 1.0  # 一阶序列边
        elif (u, v) in second_order_set:
            edge_feat[i, 1] = 1.0  # 二阶序列边
        else:
            edge_feat[i, 2] = 1.0  # 空间边
    
    # 2. 计算距离并归一化（使用14Å作为最大值）
    u_nodes = edge_index[0]  # 所有边的起点
    v_nodes = edge_index[1]  # 所有边的终点
    pos_u = pos[u_nodes]     # [E, 3] 起点坐标
    pos_v = pos[v_nodes]     # [E, 3] 终点坐标
    distances = torch.norm(pos_u - pos_v, dim=1)  # [E] 计算每条边的距离
    
    # 归一化：使用14Å作为最大值（与构图阈值一致）
    MAX_DIST = 14.0  # 构图时的半径阈值
    normalized_distances = torch.clamp(distances / MAX_DIST, max=1.0)  # 限制最大为1
    
    # 将归一化后的距离存入第4维
    edge_feat[:, 3] = normalized_distances
    
    return edge_feat

# 根据坐标构图 samples → PyG Data 列表（修改版）
def build_pyg_data_list(samples, radius=14.0):  # 修改默认半径为14
    data_list = []

    for idx, (struct_feat, seq_feat, node_labels, coords) in enumerate(samples):
        struct_feat = torch.tensor(struct_feat, dtype=torch.float)
        seq_feat    = torch.tensor(seq_feat, dtype=torch.float)
        num_nodes = struct_feat.shape[0]
        labels      = parse_node_labels(node_labels, num_nodes)
        y           = torch.tensor(labels, dtype=torch.long)
        pos         = torch.tensor(coords, dtype=torch.float)

        # 一阶序列边 (i, i+1)
        edge_seq_first = build_seq_edges_first_order(num_nodes)
        # 二阶序列边 (i, i+2)
        edge_seq_second = build_seq_edges_second_order(num_nodes)
        # 空间邻接边（半径图）
        edge_spa = radius_graph(pos, r=radius, loop=False)
        
        # 合并所有边
        edge_index = torch.cat([edge_seq_first, edge_seq_second, edge_spa], dim=1)
        
        # 生成边特征
        edge_attr = build_edge_features(edge_index, pos, edge_seq_first, edge_seq_second)

        data = Data(
            struct_feat=struct_feat,   # [N, 128]
            seq_feat=seq_feat,         # [N, 1280]
            x=torch.cat([struct_feat, seq_feat], dim=-1),  # [N, 1408]
            edge_index=edge_index,     # [2, E]
            edge_attr=edge_attr,       # [E, 4] 边特征：[类型one-hot(3维) + 归一化距离(1维)]
            pos=pos,                   # [N, 3]
            y=y                        # [N] 节点级标签
        )

        data_list.append(data)

    return data_list

# 辅助函数：构建一阶序列边
def build_seq_edges_first_order(num_nodes: int):
    """构建一阶序列边 i <-> i+1"""
    row = torch.arange(num_nodes - 1, dtype=torch.long)
    col = row + 1
    edge = torch.stack([row, col], dim=0)
    return torch.cat([edge, edge.flip(0)], dim=1)

# 辅助函数：构建二阶序列边
def build_seq_edges_second_order(num_nodes: int):
    """构建二阶序列边 i <-> i+2"""
    if num_nodes <= 2:
        return torch.empty((2, 0), dtype=torch.long)
    
    row = torch.arange(num_nodes - 2, dtype=torch.long)
    col = row + 2
    edge = torch.stack([row, col], dim=0)
    return torch.cat([edge, edge.flip(0)], dim=1)

# 保存文件
def save_pyg_dataset(samples, save_path, radius=14.0):
    data_list = build_pyg_data_list(samples, radius=radius)
    print(f"data_list info:{data_list[0]},{len(data_list)}")
    print(f"边特征维度: {data_list[0].edge_attr.shape[1]}")
    print(f"边特征示例 (前5条):\n{data_list[0].edge_attr[:5]}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    save_file = save_path / "pyg_graph_datas_495_train_edge_attr_2_1280_42.pt"
    torch.save(data_list, save_file)
    print(f"数据已保存至: {save_file}")


if __name__ == "__main__":
    save_path = Path("get_graph_edge_attr\data")
    samples = feat_align()
    save_pyg_dataset(samples, save_path)