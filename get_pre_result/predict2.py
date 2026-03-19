import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

# =========================
# Model
# =========================
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, h_struct, h_seq):
        g = torch.sigmoid(self.gate(torch.cat([h_struct, h_seq], dim=-1)))
        return g * h_struct + (1.0 - g) * h_seq


class RBPGraphNetGAT(nn.Module):
    def __init__(
        self, 
        struct_dim, 
        seq_dim, 
        hidden_dim=128, 
        num_layers=3,
        heads=4,
        dropout=0.3
    ):
        super().__init__()
        
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)
        self.seq_proj    = nn.Linear(seq_dim, hidden_dim)
        
        self.fusion = GatedFusion(hidden_dim)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i < num_layers - 1:
                self.convs.append(
                    GATConv(
                        hidden_dim, 
                        hidden_dim // heads, 
                        heads=heads,
                        dropout=dropout,
                        concat=True
                    )
                )
            else:
                self.convs.append(
                    GATConv(
                        hidden_dim, 
                        hidden_dim, 
                        heads=1,
                        dropout=dropout,
                        concat=False
                    )
                )
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x_s = F.relu(self.struct_proj(data.struct_feat))
        x_q = F.relu(self.seq_proj(data.seq_feat))
        
        x = self.fusion(x_s, x_q)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, data.edge_index)
            
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = F.elu(x)
                x = self.dropout(x)
        
        return self.classifier(x).squeeze(-1)


# =========================
# Train / Eval functions
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, data.y.float())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []

    for data in loader:
        data = data.to(device)
        logits = model(data)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu())
        all_labels.append(data.y.cpu())

    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return all_probs, all_labels


# =========================
# Main
# =========================
if __name__ == "__main__":

    data_list = torch.load(
        "get_pre_result\data\pyg_495—2_enhance_datas.pt",
        weights_only=False
    )
    data_list2 =  torch.load(
        "get_graph_data\data\pyg_graph_datas_117_test.pt",
        weights_only=False
    )
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    n_total = len(data_list)
    n_train = int(0.7 * n_total)
    n_val   = int(0.3 * n_total)

    train_data = data_list[:n_train]
    val_data   = data_list[n_train:]
    test_data  = data_list2

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=4, shuffle=False)
    test_loader  = DataLoader(test_data, batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    struct_dim=data_list[0].struct_feat.size(1)
    seq_dim=data_list[0].seq_feat.size(1)

    model = RBPGraphNetGAT(
        struct_dim=struct_dim,
        seq_dim=seq_dim,
        hidden_dim=128,
        num_layers=3,
        heads=4,
        dropout=0.3
    ).to(device)

    def count_pos_neg(val_data):
        all_labels = []
        for data in val_data:
            all_labels.append(data.y.cpu())
        all_labels = torch.cat(all_labels)
        pos_num = (all_labels == 1).sum().item()
        neg_num = (all_labels == 0).sum().item()
        return pos_num, neg_num

    pos_num, neg_num = count_pos_neg(val_data)
    pos_weight = neg_num / pos_num if pos_num > 0 else 1.0
    print(f"训练集正样本数：{pos_num} | 负样本数：{neg_num} | 正样本权重：{pos_weight:.2f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))

    best_val_mcc = -1.0
    best_val_auc = 0.0
    counter = 0

    for epoch in range(1, 51):

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_probs, val_labels = eval_epoch(
            model, val_loader, device
        )

        val_auc = roc_auc_score(val_labels, val_probs)
        val_ap  = average_precision_score(val_labels, val_probs)
        val_preds = (val_probs >= 0.5).astype(int)
        val_mcc = matthews_corrcoef(val_labels, val_preds)

        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_val_auc = val_auc
            counter = 0
            torch.save(model.state_dict(), "get_pre_result/data/best_model_gat.pt")
        else:
            counter += 1

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val AUC {val_auc:.4f} | "
            f"Val AP {val_ap:.4f} | "
            f"Val MCC {val_mcc:.4f}"
        )

    # =========================
    # Test
    # =========================
    model.load_state_dict(torch.load("get_pre_result/data/best_model_gat.pt"))
    test_probs, test_labels = eval_epoch(
        model, test_loader, device
    )

    test_roc_auc = roc_auc_score(test_labels, test_probs)
    test_ap  = average_precision_score(test_labels, test_probs)
    test_preds = (test_probs >= 0.5).astype(int)
    test_mcc = matthews_corrcoef(test_labels, test_preds)
    test_pr_auc = average_precision_score(test_labels, test_probs)

    print(
        f"Test ROC_AUC {test_roc_auc:.4f} | "
        f"Test PR_AUC {test_pr_auc:.4f} | "
        f"Test AP {test_ap:.4f} | "
        f"Test MCC {test_mcc:.4f}"
    )
