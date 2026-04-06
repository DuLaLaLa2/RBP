import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------
# 跨模态交叉注意力特征融合模块
# ------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, 
                 struct_dim=42,       # 你的结构特征维度
                 seq_dim=1280,        # 你的ESM2特征维度
                 hidden_dim=256,      # 中间映射维度（可调）
                 num_heads=4,         # 多头注意力头数
                 out_dim=256          # 最终融合输出维度
                 ):
        super().__init__()
        
        # 1. 先把两个模态映射到相同维度
        self.proj_struct = nn.Sequential(
            nn.Linear(struct_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.proj_seq = nn.Sequential(
            nn.Linear(seq_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 2. 多头交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 3. 输出投影层
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, struct_feat, seq_feat):
        """
        输入：
            struct_feat: [N, 42]
            seq_feat: [N, 1280]
        输出：
            fused_feat: [N, out_dim]  融合后的优质特征
        """
        # 维度统一映射
        s = self.proj_struct(struct_feat).unsqueeze(0)  # [1, N, D]
        t = self.proj_seq(seq_feat).unsqueeze(0)        # [1, N, D]

        # ===================== 核心：交叉注意力 =====================
        # Q = 结构特征
        # K/V = 序列特征
        # 让结构信息去“对齐/关注”关键的序列区域
        attn_output, _ = self.cross_attn(query=s, key=t, value=t)
        
        # 残差连接 + 输出
        fused = attn_output + s  # 残差
        fused = fused.squeeze(0)
        fused_feat = self.out_proj(fused)
        
        return fused_feat
