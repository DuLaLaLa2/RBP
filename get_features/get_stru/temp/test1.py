'''
# 检查路径
from pathlib import Path
di = Path("get_features\get_stru\data")
print(di)


from Bio.PDB import PDBParser
import urllib.request
# 下载 PDB文件
urllib.request.urlretrieve("https://files.rcsb.org/download/1BMV.pdb", "1bmv.pdb")
parser = PDBParser(QUIET=True)
structure = parser.get_structure("1bmv", "1bmv.pdb")
chains = set()
for model in structure:
    for chain in model:
        chains.add(chain.id)
print("实际存在的链:", sorted(chains))
'''
# s = "0000001111000000"
# print(s.count("1"))
# import freesasa

# print(freesasa.__file__)
# print(freesasa.__version__)
# hasattr(freesasa.Result, "residueResults")
# print(freesasa.__version__)  # 应输出 '2.2.1'
# print(hasattr(freesasa, 'readString'))  # 应输出 True
# print(hasattr(freesasa, 'readPdb'))     # 应输出 True

# s1 = "qwer"
# s2 = "2"
# print(s1+"_"+s2)

import numpy as np

f = np.load("get_features\get_stru\data\stru_feat\   1_3pla_L_struct.npz")
print(f.files)