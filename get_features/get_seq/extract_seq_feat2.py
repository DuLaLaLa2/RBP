import os
import torch
import esm
import numpy as np
from typing import List, Tuple

# ==============================
# 1. 解析四行一组的数据
# ==============================
def parse_four_line_format(file_path: str) -> List[Tuple[str, str]]:
    """
    解析格式：
        >3b0v_D
        MKTVRQ...
        000111...
        000111...
    返回 [(id, seq), ...]
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    assert len(lines) % 3 == 0, "文件行数不是4的倍数！"
    data = []
    for i in range(0, len(lines), 3):
        header = lines[i]
        seq = lines[i + 1]
        assert header.startswith('>'), f"第 {i+1} 行应以 '>' 开头，实际: {header}"
        protein_id = header[1:]  # 去掉 '>'
        data.append((protein_id, seq.upper()))
    return data

# ==============================
# 2. 滑动窗口提取 ESM-2 嵌入
# ==============================
def extract_esm2_embeddings_sliding_window(
    sequences: List[Tuple[str, str]],
    model,
    alphabet,
    device,
    window_size: int = 1022,
    overlap: int = 200,
    batch_size: int = 8,
):
    """
    返回 dict: {protein_id: { "seq": str, "seq_feat": np.ndarray }}
    """
    assert window_size <= 1022
    step = window_size - overlap
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    results = {}

    for protein_id, seq in sequences:
        L = len(seq)
        if L <= window_size:
            # 短序列：直接处理
            _, _, tokens = batch_converter([(protein_id, seq)])
            with torch.no_grad():
                out = model(tokens.to(device), repr_layers=[33])
            emb = out["representations"][33][0, 1:L+1].cpu().numpy()
            results[protein_id] = {"seq": seq, "seq_feat": emb}
            continue

        # 超长序列：滑动窗口
        windows, starts = [], []
        for start in range(0, L, step):
            end = min(start + window_size, L)
            if end - start < 10:
                continue
            windows.append(seq[start:end])
            starts.append(start)

        # 批量推理所有窗口
        all_embs = []
        for i in range(0, len(windows), batch_size):
            batch_data = [(f"tmp_{j}", w) for j, w in enumerate(windows[i:i+batch_size])]
            _, _, tokens = batch_converter(batch_data)
            with torch.no_grad():
                out = model(tokens.to(device), repr_layers=[33])
            reprs = out["representations"][33]
            for k in range(len(batch_data)):
                w_len = len(windows[i + k])
                emb = reprs[k, 1:w_len+1].cpu().numpy()
                all_embs.append(emb)

        # 加权融合
        D = all_embs[0].shape[1]
        total_emb = np.zeros((L, D), dtype=np.float32)
        weight_sum = np.zeros(L, dtype=np.float32)

        for start, emb in zip(starts, all_embs):
            w_len = emb.shape[0]
            end = start + w_len
            weights = np.ones(w_len, dtype=np.float32)
            ramp = min(overlap // 2, w_len // 2)
            if ramp > 0:
                weights[:ramp] = np.linspace(0.1, 1.0, ramp)
                weights[-ramp:] = np.linspace(1.0, 0.1, ramp)
            total_emb[start:end] += emb * weights[:, None]
            weight_sum[start:end] += weights

        weight_sum[weight_sum == 0] = 1
        fused_emb = total_emb / weight_sum[:, None]
        results[protein_id] = {"seq": seq, "seq_feat": fused_emb}

    return results

# ==============================
# 3. 主程序
# ==============================
def main():
    # === 配置 ===
    input_file = "get_features/get_stru/data/raw_datas/RNA-117_Test.txt"
    output_dir = "get_features/get_seq/data/seq_feat2"
    # os.makedirs(output_dir, exist_ok=True)# 目录存在则不触发异常

    # === 检查输入文件 ===
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    # === 读取数据 ===
    print("正在解析输入文件...")
    data = parse_four_line_format(input_file)
    print(f"成功加载 {len(data)} 条蛋白序列。")

    # === 加载模型 ===
    print("正在加载 ESM-2 模型...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"使用设备: {device}")

    # === 提取特征 ===
    print("正在提取 ESM-2 序列特征（含滑动窗口处理超长序列）...")
    embeddings_dict = extract_esm2_embeddings_sliding_window(
        data,
        model,
        alphabet,
        device,
        window_size=1022,
        overlap=200,
        batch_size=8
    )

    # === 保存为 .npz 文件 ===
    print("正在保存特征...")
    count=0
    id_list = []
    for full_id, result in embeddings_dict.items():
        # 分离 pdb_id 和 chain (假设格式为 "3b0v_D" 或 "1abc_A")
        if "_" in full_id and full_id.count("_") >= 1:
            parts = full_id.split("_")
            pdb_id = "_".join(parts[:-1])  # 支持如 "4xkl_fab_H"
            chain = parts[-1]
        else:
            pdb_id = full_id
            chain = "A"
        
        if full_id not in id_list:
            id_list.append(full_id)
            count +=1
            save_name = f"{count:4d}_{pdb_id}_{chain}.npz"
            save_path = os.path.join(output_dir, save_name)

            np.savez_compressed(
                save_path,
                pdb_id=pdb_id,
                chain=chain,
                seq=result["seq"],
                seq_feat=result["seq_feat"]  # shape: (L, 1280)
            )
        else:
            print(f"当前 {full_id} 蛋白链重复保存")
            
        print(f"======>{count}: {pdb_id}蛋白特征已保存，共含{len(result['seq'])}个氨基酸序列, 特征形状：{result['seq_feat'].shape}")

    print(f"✅ 全部完成！共保存 {len(embeddings_dict)} 个 .npz 文件到 '{output_dir}/'")

if __name__ == "__main__":
    main()