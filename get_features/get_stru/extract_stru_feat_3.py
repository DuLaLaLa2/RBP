# env python3.10, torch2.7.0+cu128
# -*- coding: utf-8 -*-

"""
思路：序列->pdb结构->基于氨基酸坐标获取结构特征（42-d）->GAT（128-d）
输入: RNA-495_Train.txt（含 >pdb_chain、序列、标签）
输出: 每个蛋白链的 struct_feat (L, 128)，保存为 .npz
"""

import sys
import urllib.request
from pathlib import Path
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.Align import PairwiseAligner
import traceback

import get_42d_feat
import get_128d_feat
# ==================== 配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("get_features\get_stru\data")
PDB_DIR = DATA_DIR / "pdb2"
FEAT_DIR = DATA_DIR / "stru_feat2" 
TRAIN_FILE = DATA_DIR / "raw_datas" / "RNA-117_Test.txt"

PDB_DIR.mkdir(parents=True, exist_ok=True)
FEAT_DIR.mkdir(parents=True, exist_ok=True)
loading_numb = 0 # 记录当前处理蛋白链是第几个

# 氨基酸映射
D3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}
ALLOWED_RES = set(D3TO1.keys())


# ==================== 1、解析数据集文件 ====================
def parse_train_file(file_path: str) -> list:
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    seq_info = []
    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            header = lines[i][1:]
            if "_" not in header:
                i += 3
                continue
            # 提取文本信息【蛋白id，链id，序列，标签】
            pdb_id, chain = header.split("_", 1)
            seq = lines[i + 1]
            label = lines[i + 2]
            label_1 = label.count("1")
            label_0 = label.count("0")
            seq_info.append({"pdb_id": pdb_id.lower(), "chain": chain, "seq": seq, "label":label, "label_0":label_0, "label_1":label_1})
            i += 3
        else:
            i += 1
    return seq_info

# ==================== parse_pdb_to_residues 的工具函数 ====================
def get_chain_auto(structure, chain_id):
    """
    支持：
    - chain_id = auth_asym_id
    - chain_id = label_asym_id
    """
    model = structure[0]

    # 1️⃣ 先直接按 label 查（最快）
    if chain_id in model:
        return model[chain_id]

    # 2️⃣ 再遍历查 auth_asym_id
    for chain in model:
        auth_id = chain.xtra.get("auth_asym_id")
        if auth_id == chain_id:
            return chain

    raise KeyError(f"找不到链 ID (auth 或 label): {chain_id}")


# ==================== 根据pdb文件解析残基信息 ====================
def parse_pdb_to_residues(pdb_file: str, target_chain: str) -> dict:
    '''按数据集中蛋白序列（pdb_file和target_chain）解析出每个氨基酸的：resname_1, N, CA, C, CB'''
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_file)
    residues = {}
    
    for model in structure:
        for chain in model:
            chain = get_chain_auto(structure, chain.id)# 确定找到能与序列相匹配的链
            if chain.id != target_chain:
                continue
            for residue in chain:
                hetflag, resseq, icode = residue.get_id()
                # if hetflag != " " or residue.get_resname() not in ALLOWED_RES:
                #     continue
                resname3 = residue.get_resname()

                # === 自动修复 UNK → ALA（仅当原子组成符合 ALA） ===
                if resname3 == "UNK":
                    atom_names = set(a.get_name() for a in residue.get_atoms())
                    if atom_names.issuperset({"N", "CA", "C", "O"}) and \
                    atom_names.issubset({"N", "CA", "C", "O", "CB"}):
                        resname3 = "ALA"   # ✅ 安全修复
                    else:
                        continue           # 真·UNK，跳过

                # 非标准氨基酸仍然跳过
                if hetflag != " " or resname3 not in ALLOWED_RES:
                    continue


                # 获取 CA —— 这是唯一必需的原子
                if "CA" not in residue:
                    continue  # 没有 CA，视为不存在
                
                CA = residue["CA"].get_coord()
                
                # 尝试获取其他原子，缺失则设为 None 或回退
                try:
                    N = residue["N"].get_coord()
                except KeyError:
                    N = None
                
                try:
                    C = residue["C"].get_coord()
                except KeyError:
                    C = None
                
                # CB：如果没有，用 CA 代替（常见做法）
                if "CB" in residue:
                    CB = residue["CB"].get_coord()
                else:
                    CB = CA  # 回退到 CA

                # 构建 res_id（考虑插入码）
                if icode == " ":
                    res_id = f"{chain.id}_{resseq}"
                else:
                    res_id = f"{chain.id}_{resseq}_{icode}"

                resname_1 = D3TO1[resname3]
                residues[res_id] = (resname_1, N, CA, C, CB)
    
    return residues


# ==================== 构建残基图 ====================
def build_protein_graph(node_feat: np.ndarray, ca_coords: np.ndarray, cutoff: float = 10.0) -> tuple:
    L = node_feat.shape[0]
    if L == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 4), dtype=np.float32)
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    src, dst = np.where((dist > 0) & (dist < cutoff))
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    edge_dist = dist[src, dst]
    edge_vec = diff[src, dst] / (edge_dist[:, None] + 1e-8)
    edge_attr = np.concatenate([edge_dist[:, None], edge_vec], axis=1)
    return edge_index, edge_attr


# ==================== process_single_protein 的工具函数 ====================
def fetch_af_structure(pdb_id: str, seq: str) -> Path:
    """使用 ESMFold（也可以使用本地 AlphaFold，但是比较复杂）生成结构，返回 PDB 路径"""

    af_file = PDB_DIR / f"{pdb_id}.pdb"
    if af_file.exists():
        return af_file

    try:
        # 使用 ESMFold API（免费、快速）
        print(f"尝试使用ESMFold")
        import requests
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.post(
            "https://api.esmatlas.com/foldSequence/v1/pdb/",
            data=seq,
            headers=headers,
            timeout=60
        )
        response.raise_for_status()
        with open(af_file, "w") as f:
            f.write(response.text)

        return af_file

    except Exception as e:
        print(f"⚠️ AlphaFold (ESMFold) 失败 {pdb_id}: {e}")
        return None


def align_and_extract(residues: dict, seq_fasta: str):
    '''对齐数据集氨基酸序列与pdb中残基序列'''
    if not residues:
        return {}
    
    # 构建 res_id 列表（支持插入码）
    def sort_key(rid):
        parts = rid.split('_')
        resseq = int(parts[1])
        icode = parts[2] if len(parts) > 2 else ' '
        return (resseq, icode)
    
    res_ids = sorted(residues.keys(), key=sort_key)
    seq_pdb = "".join(residues[rid][0] for rid in res_ids)# pdb中的残基序列（有确实/有插入）

    # 比对
    aligner = PairwiseAligner()
    aligner.mode = 'global'# 全长对齐
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    alignments = aligner.align(seq_fasta, seq_pdb)
    best = alignments[0]

    residues_sub = {}
    coverage = [False] * len(seq_fasta)
    fasta_idx = 0
    pdb_idx = 0
    for aa_f, aa_p in zip(best.target, best.query):
        if aa_f != "-" and aa_p != "-":
            rid = res_ids[pdb_idx]
            residues_sub[rid] = residues[rid]
            coverage[fasta_idx] = True
            fasta_idx += 1
            pdb_idx += 1
        elif aa_f != "-":
            # FASTA 有，PDB 缺失
            fasta_idx += 1
        elif aa_p != "-":
            # PDB 插入
            pdb_idx += 1

    return residues_sub, coverage


# ==================== 单蛋白处理 ====================
def process_single_protein(entry: dict) -> dict:
    
    pdb_id = entry["pdb_id"]
    chain = entry["chain"]
    seq_fasta = entry["seq"]
    num = entry["label_0"]+entry["label_1"]
    label = entry["label"]
    pdb_file = PDB_DIR / f"{pdb_id}.pdb"
    global loading_numb
    loading_numb += 1
    print(f"[{loading_numb}] ===> 当前正在处理蛋白{pdb_id}_{chain}共有{num}个氨基酸:{pdb_file}......")

    # 1、尝试下载当前蛋白链的PDB文件
    if not pdb_file.exists():
        try:
            url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
            urllib.request.urlretrieve(url, pdb_file)
            print(f"1、 PDB 文件下载成功 {pdb_id}")

        except Exception as e:
            print(f"1、 PDB 下载失败 {pdb_id}: {e}")
            pdb_file = None


    # 2、获取并解析结构（优先 PDB，失败则用 ESMFold）
    structure_file = None
    if pdb_file and pdb_file.exists():
        structure_file = pdb_file

    if not structure_file or structure_file.stat().st_size == 0:
        af_file = fetch_af_structure(pdb_id, seq_fasta)

        if af_file:
            structure_file = af_file
            chain = "A"  # ESMFold 输出通常为链 A

    if not structure_file or not structure_file.exists():
        print(f"❌process_single_protein: 无结构可用 {pdb_id}")
        return {"success": False}

    # 3、提取残基并比对
    try:
        residues = parse_pdb_to_residues(str(structure_file), chain)
        residues_matched, covered = align_and_extract(residues, seq_fasta)
        if not residues_matched:
            print(f"⚠️ 无匹配残基 {pdb_id}_{chain}")
            return {"success": False}


        # 打印覆盖摘要
        total = len(seq_fasta)
        found = sum(covered)
        missing_positions = [i + 1 for i, c in enumerate(covered) if not c]  # 1-indexed
        print(f"🧬 FASTA 长度: {total} | 有坐标的残基数: {found} ({found/total:.1%})")

        if missing_positions:
            # 控制输出长度，避免太长
            missing_str = ", ".join(map(str, missing_positions[:20]))
            if len(missing_positions) > 20:
                missing_str += f", ... (+{len(missing_positions)-20} more)"
            print(f"❌ 未找到坐标的 FASTA 位置（1-indexed）: {missing_str}")

        else:
            print("✅ 所有 FASTA 残基均找到结构坐标！")


        node_feat, _, ca_coords = get_42d_feat.feat_42d(residues_matched, pdb_file)
        edge_index, edge_attr = build_protein_graph(node_feat, ca_coords, cutoff=10.0)

        graph = {
            "id": f"{pdb_id}_{chain}",
            "x": torch.tensor(node_feat, dtype=torch.float32),
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "edge_attr": torch.tensor(edge_attr, dtype=torch.float32)
        }

        result = {
            "pdb_id"        : pdb_id,
            "chain"         : chain,
            "seq"           : seq_fasta,
            "label"         : label,
            "coords"        : ca_coords,
            "success"       : True
        }

        return result, graph
    
    except Exception as e:
        print(f"❌process_single_protein: 处理失败 {pdb_id}_{chain}: {e}")
        traceback.print_exc()
        return {"success": False}


# ==================== 主函数 ====================
def main():
    if not TRAIN_FILE.exists():
        print(f" 训练文件不存在，请重新核对文件存放路径: {TRAIN_FILE}")
        sys.exit(1)

    entries = parse_train_file(str(TRAIN_FILE))
    print(f" 任务：共 {len(entries)} 个蛋白链待处理") # 495

    success_count = 0 # 成功个数
    all_count = 0 # 计数
    all_graphs = [] # 获取所有蛋白结构图
    results = [] # 存放需要保存的结果
    
    for entry in entries:
        all_count += 1
        result, single_protein_graph= process_single_protein(entry)

        if result["success"]:
            success_count += 1
            all_graphs.append(single_protein_graph)
            results.append(result)
            # print(f" 【{all_count}】 蛋白{entry['pdb_id']}_{entry['chain']}特征提取成功")
        else:
            print(f" 【{all_count}】 蛋白{entry['pdb_id']}_{entry['chain']}特征提取失败!!!")
    
    
    GRAPH_CACHE =  "get_features\\get_stru\\data\\all_graphs2.pt"
    RESULT_CACHE =  "get_features\\get_stru\\data\\results2.pt"
    torch.save(all_graphs, GRAPH_CACHE)
    torch.save(results, RESULT_CACHE)

    print("="*40)
    print(f"=============== 成功处理 {success_count}/{len(entries)} 个============================")
    print("="*40)


if __name__ == "__main__":
    main()
    