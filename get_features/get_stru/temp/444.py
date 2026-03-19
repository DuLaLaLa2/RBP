import os
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.Align import PairwiseAligner

# =========================
# Step 1: 读取 TXT 文件
# =========================

def read_protein_txt(txt_path):
    samples = []

    with open(txt_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    assert len(lines) % 4 == 0, "TXT 文件格式错误（不是 4 的倍数）"

    for i in range(0, len(lines), 4):
        header = lines[i]
        seq = lines[i + 1]

        tag = header[1:]
        pdb_id, chain_id = tag.split("_")

        samples.append({
            "pdb_id": pdb_id.lower(),
            "chain_id": chain_id,   # ✅ 保留原样，可能是 auth 也可能是 label
            "sequence": seq
        })

    print(f"[INFO] 共读取 {len(samples)} 条蛋白链\n")
    return samples

# =========================
# Step 2: 读取 PDB
# =========================

class StructureManager:
    def __init__(self, pdb_dir="./pdb_files"):
        self.pdb_dir = pdb_dir
        self.parser = PDBParser(QUIET=True)

    def load_structure(self, pdb_id):
        path = os.path.join(self.pdb_dir, f"{pdb_id}.pdb")
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到 PDB 文件: {path}")
        return self.parser.get_structure(pdb_id, path)

# =========================
# ✅ Step 3: 链 ID 自动映射（核心升级）
# =========================

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

# =========================
# Step 4: 序列 – 结构对齐
# =========================

def align_structure_to_sequence(structure, chain_id, sequence):
    chain = get_chain_auto(structure, chain_id)

    residues = [
        r for r in chain
        if is_aa(r) and "CA" in r
    ]

    if len(residues) == 0:
        raise ValueError("structure chain has zero valid amino acids")

    pdb_seq = ""
    valid_residues = []

    for r in residues:
        resname = r.get_resname().upper()
        if resname in protein_letters_3to1:
            pdb_seq += protein_letters_3to1[resname]
        else:
            pdb_seq += "X"
        valid_residues.append(r)

    aligner = PairwiseAligner(
        mode="global",
        match_score=2,
        mismatch_score=-1,
        open_gap_score=-10,
        extend_gap_score=-1,
    )

    aln = aligner.align(sequence, pdb_seq)[0]

    mapped = [None] * len(sequence)

    t_i = p_i = 0
    for a, b in zip(aln[0], aln[1]):
        if a != "-":
            if b != "-":
                mapped[t_i] = valid_residues[p_i]
                p_i += 1
            t_i += 1
        elif b != "-":
            p_i += 1

    return mapped

# =========================
# Step 5: 打印 & 统计
# =========================

def debug_alignment(sample, structure):
    seq = sample["sequence"]
    chain_id = sample["chain_id"]

    mapped = align_structure_to_sequence(structure, chain_id, seq)

    L = len(seq)
    n_struct = sum(r is not None for r in mapped)
    cov = n_struct / L

    print(f"PDB {sample['pdb_id']}  Chain {chain_id}")
    print(f"  序列长度: {L}")
    print(f"  有结构残基: {n_struct}")
    print(f"  覆盖率: {cov:.2%}")

    return cov



# =========================
# Main
# =========================

def main():
    txt_path = "get_features\\get_stru\\data\\raw_datas\\RNA-495_Train.txt"
    pdb_dir = "get_features\\get_stru\\data\pdb"

    samples = read_protein_txt(txt_path)
    manager = StructureManager(pdb_dir)

    stats = []
    flag = 0
    for s in samples:
        flag+=1
        print(f"{flag}")
        try:
            structure = manager.load_structure(s["pdb_id"])
            cov = debug_alignment(s, structure)
            if cov != 1.0:    
                stats.append((s["pdb_id"], s["chain_id"], cov))
        except Exception as e:
            print(f"PDB {s['pdb_id']} Chain {s['chain_id']} ❌ 失败: {e}\n")
            stats.append((s["pdb_id"], s["chain_id"], None))

    print("\n========== 不完美覆盖率情况 ==========")
    for item in stats:
        print(item)

if __name__ == "__main__":
    main()