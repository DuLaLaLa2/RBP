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
import torch.nn.functional as F
from Bio.PDB import PDBParser
from tqdm import tqdm
from torch_geometric.nn import GATv2Conv
import freesasa
from Bio.Align import PairwiseAligner
import traceback
import get_42d_feat
import get_128d_feat
# ==================== 配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("get_features\get_stru\data")
PDB_DIR = DATA_DIR / "pdb"
FEAT_DIR = DATA_DIR / "stru_feat" 
TRAIN_FILE = DATA_DIR / "raw_datas" / "RNA-495_Train.txt"

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
# 理化性质
PHY_TABLE = {
    "A": [89.1, 0, 0, 1.8, 88.6], "C": [121.2, 0, 8.3, 2.5, 108.5],
    "D": [133.1, -1, 3.9, -3.5, 111.1], "E": [147.1, -1, 4.2, -3.5, 138.4],
    "F": [165.2, 0, 0, 2.8, 189.9], "G": [75.1, 0, 0, -0.4, 60.1],
    "H": [155.2, 0, 6.0, -3.2, 153.2], "I": [131.2, 0, 0, 4.5, 166.7],
    "K": [146.2, 1, 10.5, -3.9, 168.6], "L": [131.2, 0, 0, 3.8, 166.7],
    "M": [149.2, 0, 0, 1.9, 162.9], "N": [132.1, 0, 0, -3.5, 114.1],
    "P": [115.1, 0, 0, -1.6, 112.7], "Q": [146.1, 0, 0, -3.5, 143.8],
    "R": [174.2, 1, 12.5, -4.5, 173.4], "S": [105.1, 0, 0, -0.8, 89.0],
    "T": [119.1, 0, 0, -0.7, 116.1], "V": [117.1, 0, 0, 4.2, 140.0],
    "W": [204.2, 0, 0, -0.9, 227.8], "Y": [181.2, 0, 0, -1.3, 193.6],
}
# 二级结构表
SS_PROP = {
    "A": [1.45, 0.97, 0.65, 0.57], "C": [0.79, 1.30, 0.81, 0.89],
    "D": [0.98, 0.80, 1.27, 1.09], "E": [1.53, 0.26, 0.74, 1.17],
    "F": [1.12, 1.28, 0.72, 0.69], "G": [0.53, 0.81, 1.11, 1.64],
    "H": [1.24, 0.71, 0.97, 0.80], "I": [1.00, 1.60, 0.47, 0.55],
    "K": [1.07, 0.74, 0.98, 1.22], "L": [1.34, 1.22, 0.57, 0.59],
    "M": [1.20, 1.67, 0.52, 0.62], "N": [0.76, 0.48, 1.28, 1.34],
    "P": [0.59, 0.62, 1.91, 1.55], "Q": [1.00, 1.11, 0.72, 1.07],
    "R": [1.05, 0.93, 0.81, 1.05], "S": [0.97, 0.93, 1.16, 1.06],
    "T": [0.96, 1.19, 0.78, 0.97], "V": [1.14, 1.65, 0.50, 0.59],
    "W": [1.05, 1.05, 0.88, 0.82], "Y": [0.90, 1.42, 0.80, 0.88],
}

MAX_ASA_TIEN = {
    'A': 121.0, 'R': 265.0, 'N': 187.0, 'D': 187.0,
    'C': 148.0, 'Q': 214.0, 'E': 214.0, 'G': 97.0,
    'H': 216.0, 'I': 197.0, 'L': 197.0, 'K': 230.0,
    'M': 216.0, 'F': 228.0, 'P': 154.0, 'S': 143.0,
    'T': 163.0, 'W': 267.0, 'Y': 257.0, 'V': 163.0
}

def residues_to_pdb_string(residues: dict) -> str:
    """将 residues 字典转为简化 PDB 字符串（仅 N, CA, C, CB）"""
    pdb_lines = []
    atom_serial = 1
    for res_id, (resname_1, N, CA, C, CB) in residues.items():
        chain, resseq_str = res_id.split('_', 1)
        resseq = int(resseq_str)
        resname_3 = [k for k, v in D3TO1.items() if v == resname_1][0]
        
        # 写入 N, CA, C, CB（CB 可能等于 CA）
        for name, coord in [("N", N), ("CA", CA), ("C", C), ("CB", CB)]:
            x, y, z = coord
            line = f"ATOM  {atom_serial:4d}  {name:<3s}{resname_3} {chain}{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]}  "
            pdb_lines.append(line)
            atom_serial += 1
    pdb_lines.append("END")
    return "\n".join(pdb_lines)


# ==================== 解析训练文件 ====================
def parse_train_file(file_path: str) -> list:
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    seq_info = []
    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            header = lines[i][1:]
            if "_" not in header:
                i += 4
                continue
            # 提取文本信息【蛋白id，链id，序列，标签】
            pdb_id, chain = header.split("_", 1)
            seq = lines[i + 1]
            label = lines[i + 2]
            label_1 = label.count("1")
            label_0 = label.count("0")
            seq_info.append({"pdb_id": pdb_id.lower(), "chain": chain, "seq": seq, "label":label, "label_0":label_0, "label_1":label_1})
            i += 4
        else:
            i += 1
    return seq_info

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
# ==================== PDB 解析（指定链）====================
def parse_pdb_to_residues(pdb_file: str, target_chain: str) -> dict:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_file)
    residues = {}
    
    for model in structure:
        for chain in model:
            chain = get_chain_auto(structure, chain.id)
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

'''
# ==================== 手工节点特征（42维）====================
def compute_handcrafted_features(residues: dict, pdb_file: str) -> tuple:
    res_ids = sorted(residues.keys(), key=lambda x: int(x.split('_')[1]))
    L = len(res_ids)
    if L == 0:
        return np.empty((0, 42), dtype=np.float32), [], np.empty((0, 3))
    
    ca_coords = np.stack([residues[rid][2] for rid in res_ids])
    node_feat = np.zeros((L, 42), dtype=np.float32)

    # # pdb_str = residues_to_pdb_string(residues)
    # try:
    #     structure_fs = freesasa.Structure(str(pdb_file))
    #     result = freesasa.calc(structure_fs)
    #     rsa_cache = []
    #     for j in range(L):
    #         resname_1 = residues[res_ids[j]][0]
    #         try:
    #             asa = result.residueResults()[j].total
    #         except Exception:
    #             asa = 0.0
    #         max_asa = MAX_ASA_TIEN.get(resname_1, 100.0)
    #         rsa = min(max(asa / max_asa, 0.0), 1.0)
    #         rsa_cache.append(rsa)
    #     rsa_cache = np.array(rsa_cache, dtype=np.float32)
    # except Exception as e:
    #     print(f"⚠️ FreeSASA failed, using zeros for RSA: {e}")
    #     rsa_cache = np.zeros(L, dtype=np.float32)
    try:
        structure_fs = freesasa.Structure(str(pdb_file))
        result = freesasa.calc(structure_fs)

        # --- 建立 FreeSASA residue lookup ---
        fs_res_map = {}
        for r in result.residues.values():  # ✅ 新版本写法
            key = (r.chain, r.residueNumber)
            fs_res_map[key] = r.total

        rsa_cache = []
        for rid in res_ids:
            chain_id = rid.split("_")[0]
            resseq = int(rid.split("_")[1])

            asa = fs_res_map.get((chain_id, resseq), 0.0)
            resname_1 = residues[rid][0]
            max_asa = MAX_ASA_TIEN.get(resname_1, 100.0)

            rsa = min(max(asa / max_asa, 0.0), 1.0)
            rsa_cache.append(rsa)

        rsa_cache = np.array(rsa_cache, dtype=np.float32)

    except Exception as e:
        print(f"⚠️ FreeSASA failed, using zeros for RSA: {e}")
        rsa_cache = np.zeros(L, dtype=np.float32)


    for i, rid in enumerate(res_ids):
        resname_1, N, CA, C, CB = residues[rid]
        # 1. 主链几何 (6)
        d_NC=d_N_CB=d_C_CB=d_N_CA=d_C_CA = 0.0
        d_CA_CB = np.linalg.norm(CA - CB)
        if N is not None and C is not None:
            d_NC = np.linalg.norm(N - C)
            d_N_CB = np.linalg.norm(N - CB)
            d_C_CB = np.linalg.norm(C - CB)
            d_N_CA = np.linalg.norm(N - CA)
            d_C_CA = np.linalg.norm(C - CA)
        
        node_feat[i, 0:6] = [d_NC, d_CA_CB, d_N_CB, d_C_CB, d_N_CA, d_C_CA]

        # 2. 局部坐标系 (9)
        local_frame = np.eye(3).flatten()  # 默认: [1,0,0, 0,1,0, 0,0,1]
        if N is not None and C is not None:
            try:
                u = CA - N
                u_norm = np.linalg.norm(u)
                if u_norm > 1e-8:
                    u = u / u_norm
                else:
                    u = np.array([1.0, 0.0, 0.0])  # 退化情况

                v_temp = C - CA
                w = np.cross(u, v_temp)
                w_norm = np.linalg.norm(w)
                if w_norm > 1e-8:
                    w = w / w_norm
                else:
                    w = np.array([0.0, 0.0, 1.0])  # 退化

                v = np.cross(w, u)
                # 确保正交归一（可选）
                local_frame = np.concatenate([u, v, w])
            except Exception:
                # 万一数值异常，回退到默认
                pass

        node_feat[i, 6:15] = local_frame

        # 3. 侧链方向 (3)
        cb_vec = (CB - CA); cb_vec /= np.linalg.norm(cb_vec) + 1e-8
        node_feat[i, 15:18] = cb_vec

        # 4. 局部曲率 (3)
        im1 = max(0, i - 1)
        ip1 = min(L - 1, i + 1)
        v1 = ca_coords[im1] - CA
        v2 = ca_coords[ip1] - CA
        v1_norm = np.linalg.norm(v1) + 1e-8
        v2_norm = np.linalg.norm(v2) + 1e-8
        curve = v1 / v1_norm + v2 / v2_norm
        node_feat[i, 18:21] = curve

        # 5. 局部环境 (2)
        dists = np.linalg.norm(ca_coords - CA, axis=1)
        density = np.sum(dists < 10.0) - 1
        node_feat[i, 21:23] = [density, rsa_cache[i]]
        
        # 6. 二级结构倾向 (4)
        node_feat[i, 23:27] = SS_PROP.get(resname_1, SS_PROP["A"])

        # 7. 化学特性 (15)
        phy = PHY_TABLE.get(resname_1, PHY_TABLE["A"])
        mw, charge_raw, pka, hydrophobicity, volume = phy
        hb_donor = 1.0 if resname_1 in "KRWNQSTY" else 0.0
        hb_acceptor = 1.0 if resname_1 in "DENQSTYH" else 0.0
        aromatic = 1.0 if resname_1 in "FYWH" else 0.0
        aliphatic = 1.0 if resname_1 in "GAVLI" else 0.0
        sulfur_containing = 1.0 if resname_1 in "CM" else 0.0
        acidic = 1.0 if resname_1 in "DE" else 0.0
        basic = 1.0 if resname_1 in "KRH" else 0.0
        polar = 1.0 if resname_1 in "DENHKQRSTY" else 0.0
        small = 1.0 if resname_1 in "AGS" else 0.0
        tiny = 1.0 if resname_1 in "AG" else 0.0
        chem_feat = np.array([
            mw, pka, hydrophobicity, volume,
            charge_raw,
            hb_donor, hb_acceptor,
            aromatic, aliphatic,
            sulfur_containing, acidic, basic,
            polar, small, tiny
        ], dtype=np.float32)
        assert chem_feat.shape == (15,), f"Chem feat dim error for {resname_1}"
        node_feat[i, 27:42] = chem_feat

    return node_feat, res_ids, ca_coords
'''

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


# ==================== GAT 编码器 ====================
class StructureGATEncoder(torch.nn.Module):
    def __init__(self, node_dim=42, edge_dim=4, hidden_dim=64, out_dim=128, num_layers=3):
        super().__init__()
        self.node_emb = torch.nn.Linear(node_dim, hidden_dim)
        self.edge_emb = torch.nn.Linear(edge_dim, hidden_dim)
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=hidden_dim, dropout=0.1))
        self.out_proj = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.node_emb(x))
        e = F.relu(self.edge_emb(edge_attr))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, e))
        return self.out_proj(x)


def fetch_af_structure(pdb_id: str, seq: str) -> Path:

    """使用 ESMFold（或本地 AlphaFold）生成结构，返回 PDB 路径"""
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
    if not residues:
        return {}
    
    # 构建 res_id 列表（支持插入码）
    def sort_key(rid):
        parts = rid.split('_')
        resseq = int(parts[1])
        icode = parts[2] if len(parts) > 2 else ' '
        return (resseq, icode)
    
    res_ids = sorted(residues.keys(), key=sort_key)
    seq_pdb = "".join(residues[rid][0] for rid in res_ids)

    # 比对
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    alignments = aligner.align(seq_fasta, seq_pdb)
    best = alignments[0]

    # 安全遍历：zip 自动停在最短（其实等长，但更安全）
    residues_sub = {}
    pdb_idx = 0
    for aa_f, aa_p in zip(best.target, best.query):
        if aa_p == "-":
            # PDB 缺失
            print("pdb确实，fasta中氨基酸找不到坐标！")
            continue
        if aa_f == "-":
            # PDB 插入，跳过该残基
            pdb_idx += 1
            continue
        # 两者都有
        if pdb_idx >= len(res_ids):
            break  # 防越界兜底
        rid = res_ids[pdb_idx]
        residues_sub[rid] = residues[rid]
        pdb_idx += 1

    return residues_sub


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
    print(f"【{loading_numb}】 ===> 当前正在处理蛋白{pdb_id}_{chain}共有{num}个氨基酸:{pdb_file}......")

    # 1、尝试下载当前蛋白链 PDB文件
    if not pdb_file.exists():
        try:
            url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
            urllib.request.urlretrieve(url, pdb_file)
            print(f"1、 PDB 文件下载成功 {pdb_id}")

        except Exception as e:
            print(f"1、 PDB 下载失败 {pdb_id}: {e}")
            pdb_file = None


    # 2、获取并解析结构（优先 PDB，失败则用 AlphaFold）
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

    # 提取残基并比对
    try:
        residues = parse_pdb_to_residues(str(structure_file), chain)
        residues_matched = align_and_extract(residues, seq_fasta)
        if not residues_matched:
            print(f"⚠️ 无匹配残基 {pdb_id}_{chain}")
            return {"success": False}

        # === 新增：分析 FASTA 覆盖情况 ===
        # 重新做一次比对以获取位置映射（轻量，可接受）
        def get_fasta_coverage(residues, seq_fasta):

            if not residues:
                return [False] * len(seq_fasta)

            def sort_key(rid):
                parts = rid.split('_')
                resseq = int(parts[1])
                icode = parts[2] if len(parts) > 2 else ' '
                return (resseq, icode)

            res_ids = sorted(residues.keys(), key=sort_key)
            seq_pdb = "".join(residues[rid][0] for rid in res_ids)

            aligner = PairwiseAligner()
            aligner.mode = 'global'
            alignments = aligner.align(seq_fasta, seq_pdb)
            best = alignments[0]

            covered = [False] * len(seq_fasta)
            pdb_idx = 0

            for i, (aa_f, aa_p) in enumerate(zip(best.target, best.query)):
                if aa_p == "-" or aa_f == "-":
                    continue
                if pdb_idx < len(res_ids) and res_ids[pdb_idx] in residues_matched:
                    covered[i] = True
                pdb_idx += 1
            return covered

        covered = get_fasta_coverage(residues, seq_fasta)

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

        all_graphs = []

        graph = {
            "id": pdb_id+"_"+chain,
            "x": node_feat,                       # torch.FloatTensor (N,42)
            "edge_index": edge_index,     # torch.LongTensor (2,E)
            "edge_attr": edge_attr        # torch.FloatTensor (E,4)
        }

        all_graphs.append(graph)

        # 1️⃣ 划分数据
        pretrain_graphs, rest_graphs = get_128d_feat.split_pretrain_set(all_graphs, 0.3)

        # 2️⃣ 预训练
        encoder = get_128d_feat.pretrain_gat(pretrain_graphs, epochs=80)

        # 3️⃣ 冻结
        for p in encoder.parameters():
            p.requires_grad = False

        # 4️⃣ 给所有蛋白提特征（包括预训练用过的）
        struct_embeddings = []
        for g in all_graphs:
            z = get_128d_feat.extract_struct_embedding(encoder, g)
            struct_embeddings.append(z)

        # model = StructureGATEncoder().to(DEVICE)
        # model.eval()

        # with torch.no_grad():
        #     x = torch.from_numpy(node_feat).float().to(DEVICE)
        #     ei = torch.from_numpy(edge_index).long().to(DEVICE)
        #     ea = torch.from_numpy(edge_attr).float().to(DEVICE)
        #     struct_feat = model(x, ei, ea).cpu().numpy()

        return {
            "pdb_id"        : pdb_id,
            "chain"         : chain,
            "seq"           : seq_fasta,
            "struct_feat"   : struct_embeddings.astype(np.float32),
            "label"         : label,
            "coords"        : ca_coords,
            "success"       : True
        }
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
    # for e in entries:
    #     num = e['label_0']+e['label_1']
    #     print(f"蛋白{e['pdb_id']}中总样本数为{num}，其中少数类样本比例：{e['label_1']/num})")
    #     break

    success_count = 0
    count = 0 # 计数
    for entry in tqdm(entries, desc="Processing"):
        count += 1
        result = process_single_protein(entry)
        if result["success"]:
            # key = f"{result['pdb_id']}_{result['chain']}"
            # save_path = FEAT_DIR / f"{loading_numb:04d}_{key}_struct.npz"
            # np.savez_compressed(
            #     save_path,
            #     pdb_id        =result["pdb_id"],
            #     chain         =result["chain"],
            #     seq           =result["seq"],
            #     struct_feat   =result["struct_feat"],
            #     label         =result["label"],
            #     coords        =result["coords"]
            # )
            print(f" 【{count}】 蛋白{entry['pdb_id']}_{entry['chain']}特征提取成功")
            success_count += 1
        else:
            print(f" 【{count}】 蛋白{entry['pdb_id']}_{entry['chain']}特征提取失败!!!")
    
    print("="*40)
    print(f"=============== 成功处理 {success_count}/{len(entries)} 个============================")
    print("="*40)




if __name__ == "__main__":
    main()