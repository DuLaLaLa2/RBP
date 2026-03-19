import torch
import random
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F


class StructuralGATEncoder(nn.Module):
    def __init__(self, in_dim=42, edge_dim=4, hidden_dim=64, out_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU()
        )

        self.gat1 = GATv2Conv(
            hidden_dim, hidden_dim, heads=4, concat=True,
            dropout=0.1, edge_dim=edge_dim
        )
        self.gat2 = GATv2Conv(
            hidden_dim * 4, out_dim, heads=1, concat=False,
            dropout=0.1, edge_dim=edge_dim
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.proj(x)
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        return x
    

class StructuralPretrainModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Linear(128, 42)
        # self.dist_head = nn.Linear(128, 1)

    def forward(self, x, edge_index, edge_attr):
        z = self.encoder(x, edge_index, edge_attr)
        recon = self.decoder(z)
        return z, recon
  


def split_pretrain_set(all_graphs, pretrain_ratio=0.3, seed=42):
    """
    all_graphs: list of graph dicts (一个蛋白一个)
    """
    random.seed(seed)
    idx = list(range(len(all_graphs)))
    random.shuffle(idx)

    n_pre = int(len(idx) * pretrain_ratio)
    pre_idx = idx[:n_pre]
    rest_idx = idx[n_pre:]

    pretrain_graphs = [all_graphs[i] for i in pre_idx]
    downstream_graphs = [all_graphs[i] for i in rest_idx]

    return pretrain_graphs, downstream_graphs

  
def pretrain_gat(
    graphs,
    epochs=80,
    mask_ratio=0.3,
    lr=3e-3,
    device="cuda"
):
    encoder = StructuralGATEncoder().to(device)
    model = StructuralPretrainModel(encoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    mse = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        total_loss = 0.0

        for g in graphs:
            x = g["x"].to(device)
            ei = g["edge_index"].to(device)
            ea = g["edge_attr"].to(device)

            # ---- Mask ----
            mask = torch.rand(x.size(0), device=device) < mask_ratio
            x_masked = x.clone()
            x_masked[mask] = 0.0

            z, recon = model(x_masked, ei, ea)

            # Masked feature loss
            loss_feat = mse(recon[mask], x[mask])

            # Distance loss
            zi = z[ei[0]]
            zj = z[ei[1]]
            pred_dist = torch.norm(zi - zj, dim=-1)
            true_dist = ea[:, 0]
            loss_dist = mse(pred_dist, true_dist)

            loss = loss_feat + 0.1 * loss_dist

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (ep + 1) % 10 == 0:
            print(f"[Pretrain] Epoch {ep+1:03d} | Loss {total_loss/len(graphs):.4f}")

    return encoder

def extract_struct_embedding(encoder, graph, device="cuda"):
    encoder.eval()
    with torch.no_grad():
        x = graph["x"].to(device)
        ei = graph["edge_index"].to(device)
        ea = graph["edge_attr"].to(device)
        z = encoder(x, ei, ea)
    return z.cpu()   # (N, 128)