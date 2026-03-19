def compare_sequences():
    print("请输入 FASTA 序列（粘贴后回车，空行结束）:")
    fasta_lines = []
    while True:
        line = input().strip()
        if line == "":
            break
        fasta_lines.append(line)
    seq_fasta = "".join(fasta_lines).upper()

    print("\n请输入 PDB 提取的序列（粘贴后回车，空行结束）:")
    pdb_lines = []
    while True:
        line = input().strip()
        if line == "":
            break
        pdb_lines.append(line)
    seq_pdb = "".join(pdb_lines).upper()

    print("\n" + "="*50)
    print(f"FASTA 长度: {len(seq_fasta)}")
    print(f"PDB   长度: {len(seq_pdb)}")
    print("="*50)

    if seq_fasta == seq_pdb:
        print("✅ 两段序列完全一致！")
        return

    print("\n🔍 差异位置（从1开始计数）:")
    max_len = max(len(seq_fasta), len(seq_pdb))
    diff_count = 0

    for i in range(max_len):
        aa_f = seq_fasta[i] if i < len(seq_fasta) else "-"
        aa_p = seq_pdb[i] if i < len(seq_pdb) else "-"
        if aa_f != aa_p:
            print(f"位置 {i+1:4}: FASTA={aa_f} | PDB={aa_p}")
            diff_count += 1

    print(f"\n📊 共 {diff_count} 处不同。")

if __name__ == "__main__":
    compare_sequences()