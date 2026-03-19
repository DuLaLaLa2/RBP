import torch
import esm
import numpy as np

# 加载预训练模型和对应的分词器（alphabet）
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.eval()  # 设置为评估模式（不启用 dropout 等）

# 获取批处理工具
batch_converter = alphabet.get_batch_converter()

data = [
    ("1k8w", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("2gtt", "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN")
]

# 将序列转为 batch 输入（token_ids, attention_mask 等）
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# 如果有 GPU，移到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
batch_tokens = batch_tokens.to(device)

# 前向传播，获取每层的输出
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    
# 提取最后一层的 token 表示（嵌入）
token_representations = results["representations"][33]  # shape: [B, L+2, D]

# 对每个蛋白，取中间部分（去掉首尾特殊 token）
per_residue_embeddings = {}
for i, (k, seq) in enumerate(data):
    emb = token_representations[i, 1 : len(seq) + 1].cpu().numpy()  # shape: [seq_len, hidden_dim]
    per_residue_embeddings[k] = emb
np.savez("./get_features/get_seq/data/features/protein_embeddings.npz", per_residue_embeddings)

print("蛋白嵌入特征：",per_residue_embeddings)
