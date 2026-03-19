import torch
import os
import random

# 数据路径
INPUT_PATH = r"get_graph_data\data\pyg_graph_datas_495_train.pt"
OUTPUT_DIR = r"get_protain_nodes\test\data"

TRAIN_RATIO = 0.7
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 50)
print("步骤1: 划分数据集")
print("=" * 50)

print(f"加载原始数据: {INPUT_PATH}")
all_data = torch.load(INPUT_PATH, weights_only=False, map_location='cpu')
print(f"原始数据数量: {len(all_data)}")

random.shuffle(all_data)
n_train = int(len(all_data) * TRAIN_RATIO)
n_val = len(all_data) - n_train

train_data = all_data[:n_train]
val_data = all_data[n_train:]

print(f"\n数据划分:")
print(f"  训练集: {len(train_data)}")
print(f"  验证集: {len(val_data)}")

# 保存验证集（不增强）
os.makedirs(OUTPUT_DIR, exist_ok=True)
val_path = os.path.join(OUTPUT_DIR, "val_graphs.pt")
torch.save(val_data, val_path)
print(f"\n验证集已保存: {val_path}")

# 保存训练集（用于GAN训练和增强）
train_path = os.path.join(OUTPUT_DIR, "train_graphs_for_gan.pt")
torch.save(train_data, train_path)
print(f"训练集已保存: {train_path}")

print("\n" + "=" * 50)
print("步骤2: 训练GAN")
print("=" * 50)
print("请运行: python get_protain_nodes/test/train.py")
print("(脚本会自动加载 train_graphs_for_gan.pt)")

print("\n" + "=" * 50)
print("步骤3: 增强训练集")
print("=" * 50)
print("请运行: python get_protain_nodes/test/augment.py")
print("(脚本会自动加载训练集并增强)")