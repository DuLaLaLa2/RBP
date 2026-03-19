import numpy as np
import freesasa

# ---------------- 配置与常量 ----------------
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
# 3. RSA 计算所需
MAX_ASA_TIEN = {
    'A': 121.0, 'R': 265.0, 'N': 187.0, 'D': 187.0,
    'C': 148.0, 'Q': 214.0, 'E': 214.0, 'G': 97.0,
    'H': 216.0, 'I': 197.0, 'L': 197.0, 'K': 230.0,
    'M': 216.0, 'F': 228.0, 'P': 154.0, 'S': 143.0,
    'T': 163.0, 'W': 267.0, 'Y': 257.0, 'V': 163.0
}

def feat_42d(residues: dict, pdb_file: str) -> tuple:
    res_ids = sorted(residues.keys(), key=lambda x: int(x.split('_')[1]))
    L = len(res_ids)
    if L == 0:
        return np.empty((0, 42), dtype=np.float32), [], np.empty((0, 3))
    
    ca_coords = np.stack([residues[rid][2] for rid in res_ids])
    node_feat = np.zeros((L, 42), dtype=np.float32)

    try:
        structure_fs = freesasa.Structure(str(pdb_file))
        result = freesasa.calc(structure_fs)

        # ✅ Windows pip 版唯一可用接口
        residue_areas = result.residueAreas()  
        # 结构： {chain_id: {resnum: {resname: area}}}

        rsa_cache = []
        for rid in res_ids:
            chain_id = rid.split("_")[0]
            resseq = int(rid.split("_")[1])

            asa = 0.0
            if chain_id in residue_areas and resseq in residue_areas[chain_id]:
                # residue_areas[chain][resnum] 是 dict，如 {'ALA': 45.3}
                asa = list(residue_areas[chain_id][resseq].values())[0]

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

        # ================================
        # >>> MODIFIED <<< 处理 CB 缺失（如 Gly）
        # ================================
        if CB is None:
            # 使用 N-CA-C 构造虚拟 CB（标准做法）
            if N is not None and C is not None:
                b = CA - N
                b /= np.linalg.norm(b) + 1e-8

                c = C - CA
                c /= np.linalg.norm(c) + 1e-8

                a = np.cross(b, c)
                a /= np.linalg.norm(a) + 1e-8

                CB = CA + 1.52 * (b + c + a)
            else:
                # 极端退化情况（几乎不会发生）
                CB = CA.copy()

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