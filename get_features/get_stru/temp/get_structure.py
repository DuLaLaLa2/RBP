#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
给定 PDB ID，自动下载并提取结构特征 (L,128)
用法:
    python run_pdbid.py 1vii
"""
import os, sys, urllib.request, pathlib, numpy as np, torch
from get_all_stru_fea import pdb_to_resdict, build_graph, GATEncoder

DOWNLOAD_URL = "https://files.rcsb.org/download/{pdbid}.pdb"

def download_pdb(pdbid: str, outdir: pathlib.Path):
    pdbid = pdbid.lower()
    outfile = outdir / f"{pdbid}.pdb"
    url = DOWNLOAD_URL.format(pdbid=pdbid.upper())
    urllib.request.urlretrieve(url, outfile)
    return str(outfile)

def extract(pdb_file: str, device) -> tuple[str, np.ndarray, np.ndarray]:
    model = GATEncoder().to(device)
    pid, feat, mask = extract_single(pdb_file, model, device)
    return pid, feat, mask

def extract_single(pdb_file: str, model: GATEncoder, device):
    # 封装一下，保持与原接口一致
    res_dict = pdb_to_resdict(pdb_file)
    if len(res_dict) == 0:
        return pathlib.Path(pdb_file).stem, np.empty((0, 128), dtype=np.float32), np.empty(0, dtype=bool)
    node_feat, edge_index, edge_feat = build_graph(res_dict)
    x = torch.from_numpy(node_feat).to(device)
    edge_index = torch.from_numpy(edge_index).to(device)
    edge_attr = torch.from_numpy(edge_feat).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, edge_attr)
    feat = out.cpu().numpy()
    mask = np.ones(feat.shape[0], dtype=bool)
    return pathlib.Path(pdb_file).stem, feat, mask

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python run_pdbid.py <PDB-ID>")
    pdbid = sys.argv[1].lower().strip()

    out_pdb = pathlib.Path("pdb")
    out_npz = pathlib.Path("out")
    out_pdb.mkdir(exist_ok=True)
    out_npz.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pdb_file = download_pdb(pdbid, out_pdb)
    pid, feat, mask = extract(pdb_file, device)
    print(f"PDB {pid}  长度 L={feat.shape[0]}  shape={feat.shape}")

    npz_file = out_npz / f"{pid}_struct128.npz"
    np.savez_compressed(npz_file, pid=pid, feat=feat, mask=mask)
    print(f"saved -> {npz_file}")

if __name__ == "__main__":
    main()