import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import freesasa
import warnings
from Bio import BiopythonWarning
from Bio.PDB import PDBList, PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import three_to_one, is_aa
from Bio.Align import PairwiseAligner
from torch_geometric.nn import GATv2Conv

# ---------------- 配置与常量 ----------------
warnings.simplefilter('ignore', BiopythonWarning)
freesasa.setVerbosity(freesasa.nowarnings)

# 1. 二级结构倾向表
SS_PROP = {
    'A': [1.42, 0.83, 0.66, 0.70], 'R': [0.98, 0.93, 0.95, 1.05], 'N': [0.67, 0.89, 1.56, 1.20],
    'D': [1.01, 0.54, 1.46, 1.20], 'C': [0.70, 1.19, 1.19, 1.00], 'Q': [1.11, 1.10, 0.98, 0.85],
    'E': [1.51, 0.37, 0.74, 0.85], 'G': [0.57, 0.75, 1.56, 1.50], 'H': [1.00, 0.87, 0.95, 1.00],
    'I': [1.08, 1.60, 0.47, 0.65], 'L': [1.21, 1.30, 0.59, 0.65], 'K': [1.16, 0.74, 1.01, 1.05],
    'M': [1.45, 1.05, 0.60, 0.65], 'F': [1.13, 1.38, 0.60, 0.70], 'P': [0.57, 0.55, 1.52, 2.00],
    'S': [0.77, 0.75, 1.43, 1.10], 'T': [0.83, 1.19, 0.96, 0.95], 'W': [1.08, 1.37, 0.96, 0.80],
    'Y': [0.69, 1.47, 1.14, 0.95], 'V': [1.06, 1.70, 0.50, 0.60], 'X': [1.00, 1.00, 1.00, 1.00]
}

# 2. 物理化学性质表
PHY_TABLE = {
    'A': [0.44, 0.0, 0.43, 0.36, 0.44], 'R': [0.87, 1.0, 0.89, -0.9, 0.86], 'N': [0.66, 0.0, 0.40, -0.7, 0.58],
    'D': [0.66, -1.0, 0.27, -0.7, 0.55], 'C': [0.60, 0.0, 0.58, 0.50, 0.54], 'Q': [0.73, 0.0, 0.40, -0.7, 0.72],
    'E': [0.73, -1.0, 0.30, -0.7, 0.69], 'G': [0.37, 0.0, 0.43, -0.08, 0.30], 'H': [0.77, 0.5, 0.43, -0.6, 0.76],
    'I': [0.65, 0.0, 0.43, 0.90, 0.83], 'L': [0.65, 0.0, 0.43, 0.76, 0.83], 'K': [0.73, 1.0, 0.75, -0.7, 0.84],
    'M': [0.74, 0.0, 0.43, 0.38, 0.81], 'F': [0.82, 0.0, 0.43, 0.56, 0.94], 'P': [0.57, 0.0, 0.43, -0.3, 0.56],
    'S': [0.52, 0.0, 0.43, -0.1, 0.44], 'T': [0.59, 0.0, 0.43, -0.1, 0.58], 'W': [1.02, 0.0, 0.43, -0.1, 1.14],
    'Y': [0.90, 0.0, 0.71, -0.2, 0.96], 'V': [0.58, 0.0, 0.43, 0.84, 0.70], 'X': [0.50, 0.0, 0.50, 0.00, 0.50]
}

# ================= 构图函数 (引入边特征) =================
def build_protein_graph(node_feat: np.ndarray, ca_coords: np.ndarray, cutoff: float = 10.0):
    """
    用户提供的构图逻辑：计算距离和方向向量
    Return:
        edge_index: [2, E]
        edge_attr:  [E, 4] (dist, vec_x, vec_y, vec_z)
    """
    L = node_feat.shape[0]
    if L == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 4), dtype=np.float32)
    
    # 利用 numpy 广播计算差异
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    
    # 建立连接 (排除自身 dist > 0)
    src, dst = np.where((dist > 0) & (dist < cutoff))
    
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    edge_dist = dist[src, dst]
    edge_vec = diff[src, dst] / (edge_dist[:, None] + 1e-8) # 归一化向量
    
    # 边特征: [距离, 向量x, 向量y, 向量z] -> 4维
    edge_attr = np.concatenate([edge_dist[:, None], edge_vec], axis=1).astype(np.float32)
    
    return edge_index, edge_attr

# ================= 42维特定特征计算器 =================
class SpecFeatureCalculator:
    def _get_virtual_cb(self, residue, n_vec, ca_vec, c_vec):
        if residue.get_resname() != 'GLY' and 'CB' in residue:
            return residue['CB'].get_vector().get_array()
        b, c = ca_vec - n_vec, c_vec - ca_vec
        a = np.cross(b, c)
        return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca_vec

    def compute_features(self, aligned_residues):
        L = len(aligned_residues)
        node_feat = np.zeros((L, 42), dtype=np.float32)
        
        # Pass 1: 提取坐标与SASA
        data_cache, ca_coords_all = [], []
        valid_indices_map = {} # Map full_index -> compact_index
        
        for i, res in enumerate(aligned_residues):
            if res is None:
                data_cache.append(None); continue
            try:
                N = res['N'].get_vector().get_array()
                CA = res['CA'].get_vector().get_array()
                C = res['C'].get_vector().get_array()
                CB = self._get_virtual_cb(res, N, CA, C)
                rsa = res.xtra.get('SASA_REL', 0.0)
                
                valid_indices_map[i] = len(ca_coords_all)
                ca_coords_all.append(CA)
                data_cache.append((three_to_one(res.get_resname()) if res.get_resname() in three_to_one else 'X', N, CA, C, CB, rsa))
            except: data_cache.append(None)
            
        ca_coords_np = np.array(ca_coords_all)
        if len(ca_coords_np) == 0: return node_feat, np.empty((0,3)), []

        # Pass 2: 计算特征
        valid_mask_indices = [] # 记录哪些行是有效的
        for i in range(L):
            data = data_cache[i]
            if data is None: continue
            valid_mask_indices.append(i)
            
            resname_1, N, CA, C, CB, rsa = data
            
            # --- 1. 主链几何 (6) ---
            d_vals = [np.linalg.norm(N-C), np.linalg.norm(CA-CB), np.linalg.norm(N-CB), 
                      np.linalg.norm(C-CB), np.linalg.norm(N-CA), np.linalg.norm(C-CA)]
            node_feat[i, 0:6] = d_vals

            # --- 2. 局部坐标系 (9) ---
            try:
                u = CA - N; u /= (np.linalg.norm(u) + 1e-8)
                v_tmp = C - CA
                w = np.cross(u, v_tmp); w /= (np.linalg.norm(w) + 1e-8)
                v = np.cross(w, u)
                node_feat[i, 6:15] = np.concatenate([u, v, w])
            except: node_feat[i, 6:15] = np.eye(3).flatten()

            # --- 3. 侧链方向 (3) ---
            cb_v = CB - CA; node_feat[i, 15:18] = cb_v / (np.linalg.norm(cb_v) + 1e-8)

            # --- 4. 局部曲率 (3) ---
            idx_compact = valid_indices_map[i]
            im1, ip1 = max(0, idx_compact-1), min(len(ca_coords_np)-1, idx_compact+1)
            v1, v2 = ca_coords_np[im1]-CA, ca_coords_np[ip1]-CA
            node_feat[i, 18:21] = (v1/(np.linalg.norm(v1)+1e-8)) + (v2/(np.linalg.norm(v2)+1e-8))

            # --- 5. 环境 (2) ---
            dists = np.linalg.norm(ca_coords_np - CA, axis=1)
            node_feat[i, 21:23] = [np.sum(dists < 10.0) - 1, rsa]

            # --- 6. SS (4) ---
            node_feat[i, 23:27] = SS_PROP.get(resname_1, SS_PROP["A"])

            # --- 7. Chem (15) ---
            phy = PHY_TABLE.get(resname_1, PHY_TABLE["A"])
            bools = [
                resname_1 in "KRWNQSTY", resname_1 in "DENQSTYH", resname_1 in "FYWH",
                resname_1 in "GAVLI", resname_1 in "CM", resname_1 in "DE", resname_1 in "KRH",
                resname_1 in "DENHKQRSTY", resname_1 in "AGS", resname_1 in "AG"
            ]
            node_feat[i, 27:42] = np.concatenate([phy, [float(b) for b in bools]])

        return node_feat, ca_coords_np, valid_mask_indices

# ================= 升级版 GAT (支持边特征) =================
class StructuralGATEncoder(nn.Module):
    def __init__(self, in_dim=42, edge_dim=4, hidden_dim=64, out_dim=128):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())
        
        # [关键升级] edge_dim=4 告诉 GAT 考虑边的距离和方向
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=True, dropout=0.1, edge_dim=edge_dim)
        self.gat2 = GATv2Conv(hidden_dim * 4, out_dim, heads=1, concat=False, dropout=0.1, edge_dim=edge_dim)
        
    def forward(self, x, edge_index, edge_attr):
        x = self.proj(x)
        # 传入 edge_attr
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = torch.elu(x)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        return x

class StructuralAutoEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 42))
        
    def forward(self, x, edge_index, edge_attr):
        z = self.encoder(x, edge_index, edge_attr)
        recon = self.decoder(z)
        return z, recon

def pretrain_gat_model(raw_x, edge_index, edge_attr, epochs=50):
    """带边特征的自监督训练"""
    print(f"      [训练] 启动 GATv2 预训练 (Nodes={raw_x.shape[0]}, EdgeAttr Dim={edge_attr.shape[1]})...")
    
    encoder = StructuralGATEncoder(in_dim=42, edge_dim=4, out_dim=128)
    model = StructuralAutoEncoder(encoder)
    
    if raw_x.shape[0] < 5: return encoder 
    
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        _, recon = model(raw_x, edge_index, edge_attr)
        loss = criterion(recon, raw_x)
        loss.backward()
        optimizer.step()
    
    print(f"      [完成] Loss: {loss.item():.4f}")
    return model.encoder

# ================= 流程整合 =================
class StructureManager:
    def __init__(self, download_dir="./pdb_structures"):
        self.download_dir = download_dir
        if not os.path.exists(download_dir): os.makedirs(download_dir)
        self.pdblist = PDBList(verbose=False)
    def get_structure(self, pdb_id):
        clean_id = pdb_id[:4].lower()
        cif, pdb = os.path.join(self.download_dir, f"{clean_id}.cif"), os.path.join(self.download_dir, f"pdb{clean_id}.ent")
        if os.path.exists(cif): return MMCIFParser(QUIET=True).get_structure(clean_id, cif)
        if os.path.exists(pdb): return PDBParser(QUIET=True).get_structure(clean_id, pdb)
        try: 
            f = self.pdblist.retrieve_pdb_file(clean_id, pdir=self.download_dir, file_format='mmCif')
            if os.path.exists(f): return MMCIFParser(QUIET=True).get_structure(clean_id, f)
        except: pass
        try:
            f = self.pdblist.retrieve_pdb_file(clean_id, pdir=self.download_dir, file_format='pdb')
            return PDBParser(QUIET=True).get_structure(clean_id, f)
        except: return None

def run_stage_1_structure_pipeline(target_sequence, pdb_id, chain_id='A'):
    print(f"[*] 处理任务: PDB={pdb_id} | Len={len(target_sequence)}")
    
    # 1. 解析与SASA
    mgr = StructureManager()
    structure = mgr.get_structure(pdb_id)
    if not structure: return None, None
    chain = structure[0][chain_id] if chain_id in structure[0] else None
    if not chain: return None, None
    
    try:
        rsa = freesasa.classifyResults(freesasa.calc(structure), structure)
        for r in chain: 
            if is_aa(r): r.xtra['SASA_REL'] = rsa['Residue'][r.get_segid() or chain.id][r.id[1]].relativeTotal
    except: pass

    # 2. 对齐
    valid_res = [r for r in chain if is_aa(r) and 'CA' in r]
    pdb_seq = "".join([three_to_one(r.get_resname()) if r.get_resname() in three_to_one else 'X' for r in valid_res])
    aligner = PairwiseAligner()
    aligner.mode, aligner.match_score, aligner.open_gap_score = 'global', 2, -10
    aln = aligner.align(target_sequence, pdb_seq)[0]
    
    mapped_res = [None] * len(target_sequence)
    t_ptr, p_ptr = 0, 0
    t_vec, p_vec = aln[0, :], aln[1, :]
    for i in range(len(t_vec)):
        if t_vec[i] != '-':
            if p_vec[i] != '-' and p_ptr < len(valid_res):
                mapped_res[t_ptr] = valid_res[p_ptr]; p_ptr += 1
            elif p_vec[i] != '-': p_ptr += 1
            t_ptr += 1
        elif p_vec[i] != '-': p_ptr += 1

    # 3. 特征计算
    calc = SpecFeatureCalculator()
    # node_feat_full: [L, 42] (含0), ca_coords_valid: [M, 3], valid_indices: List[M]
    node_feat_full, ca_coords_valid, valid_indices = calc.compute_features(mapped_res)

    # 4. **构图 (使用用户提供的函数)**
    # 注意：我们只对存在的节点 (Valid Nodes) 进行构图和 GAT 训练
    # 否则 GAT 无法处理缺失坐标的节点距离
    if len(valid_indices) < 2:
        print("[!] 有效结构节点太少，返回全0特征")
        return np.zeros((len(target_sequence), 128)), node_feat_full

    node_feat_valid = node_feat_full[valid_indices] # [M, 42]
    
    # 调用你的函数
    edge_index_np, edge_attr_np = build_protein_graph(node_feat_valid, ca_coords_valid, cutoff=10.0)
    
    # 转 Tensor
    x_tensor = torch.tensor(node_feat_valid) # [M, 42]
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32) # [E, 4]
    
    if edge_index.numel() == 0: # 孤立点防护
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float32)

    # 5. 自监督训练 (带边特征)
    trained_encoder = pretrain_gat_model(x_tensor, edge_index, edge_attr, epochs=50)
    
    # 6. 推理与还原
    trained_encoder.eval()
    with torch.no_grad():
        embedding_valid = trained_encoder(x_tensor, edge_index, edge_attr) # [M, 128]
    
    # 将 Valid 节点的 128维特征 映射回全长 L
    final_embedding_L = np.zeros((len(target_sequence), 128), dtype=np.float32)
    final_embedding_L[valid_indices] = embedding_valid.numpy()

    print(f"[*] 成功输出. Feature Map Reconstructed to [L={len(target_sequence)}, 128]")
    return final_embedding_L, node_feat_full

def read_protein_info(file_path):

    """
    从指定路径读取蛋白信息。
    文件中的每四行为一组数据：标识符、序列、标签、第四行暂无用处。
    返回包含字典的列表，每个字典代表一种蛋白质的信息。
    """
    proteins = []
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    for i in range(0, len(lines), 4):
        protein = {
            "id": lines[i].replace('>', ''),
            "sequence": lines[i+1],
            "label": lines[i+2],
        }
        proteins.append(protein)
        print(f"蛋白链长度为：{len(proteins)}")
    return proteins


def process_and_save_features(proteins, output_dir="get_features\\get_stru\\data\\stru_feat"):

    """
    遍历proteins列表，为每个蛋白质运行特征提取流程，并保存结果。
    """
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    for protein in proteins:
        pdb_id = protein["id"].split('_')[0]  # 假设ID的前半部分是PDB ID
        chain_id = protein["id"].split('_')[1] if '_' in protein["id"] else 'A'
        emb, node_feat_full = run_stage_1_structure_pipeline(protein["sequence"], pdb_id, chain_id=chain_id)

        # 保存特征
        np.save(os.path.join(output_dir, f"{protein['id']}_embedding.npy"), emb)
        np.save(os.path.join(output_dir, f"{protein['id']}_node_features.npy"), node_feat_full)

        
if __name__ == "__main__":
    input_file = "get_features\\get_stru\\data\\raw_datas\\RNA-495_Train.txt"
    proteins = read_protein_info(input_file)
    # 处理并保存特征
    process_and_save_features(proteins)