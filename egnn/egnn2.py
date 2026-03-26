import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
import warnings
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, f1_score, precision_score, recall_score

warnings.filterwarnings('ignore')

# ===================== EGNN 底层工具函数 =====================
def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0.0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0.0)
    count = data.new_full(result_shape, 0.0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1.0)

# ===================== E_GCL 层 =====================
class E_GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0,
                 act_fn=nn.SiLU(), residual=True, attention=False,
                 normalize=False, coords_agg='mean', tanh=False):
        super().__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.LayerNorm(hidden_nf),
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.LayerNorm(hidden_nf)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.LayerNorm(hidden_nf),
            nn.Linear(hidden_nf, output_nf)
        )

        coord_mlp = [
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.LayerNorm(hidden_nf),
            nn.Linear(hidden_nf, 1, bias=False)
        ]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid()
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if hasattr(m, 'out_features') and m.out_features == 1:
                torch.nn.init.xavier_uniform_(m.weight, gain=0.001)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr=None):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        else:
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1, keepdim=True)
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm
        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[edge_index[0]], h[edge_index[1]], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_feat

# ===================== EGNN 主干 =====================
class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0,
                 act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False,
                 normalize=False, tanh=False):
        super().__init__()
        self.embedding_in = nn.Sequential(
            nn.Linear(in_node_nf, hidden_nf),
            nn.LayerNorm(hidden_nf),
            act_fn
        )
        self.embedding_out = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.LayerNorm(hidden_nf),
            nn.Linear(hidden_nf, out_node_nf)
        )
        self.layers = nn.ModuleList([
            E_GCL(hidden_nf, hidden_nf, hidden_nf, edges_in_d=in_edge_nf,
                  act_fn=act_fn, residual=residual, attention=attention,
                  normalize=normalize, tanh=tanh)
            for _ in range(n_layers)
        ])

    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        for layer in self.layers:
            h, x, _ = layer(h, edges, x, edge_attr)
        h = self.embedding_out(h)
        return h, x

# ===================== 节点分类模型 =====================
class EGNNClassifier(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, num_classes, in_edge_nf=5, n_layers=4, attention=True):
        super().__init__()
        self.egnn = EGNN(
            in_node_nf=in_node_nf,
            hidden_nf=hidden_nf,
            out_node_nf=hidden_nf,
            in_edge_nf=in_edge_nf,
            n_layers=n_layers,
            attention=attention
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf // 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_nf // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_nf // 2, num_classes)
        )

    def forward(self, data):
        h = data.x
        pos = data.pos
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h_out, _ = self.egnn(h, pos, edge_index, edge_attr)
        logits = self.classifier(h_out)
        return logits

# ===================== 6 大指标 =====================
def compute_metrics(y_true, y_pred, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return {
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': auc(recall, precision),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred)
    }

# ===================== 训练器 =====================
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, lr=1e-3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50)
        self.scaler = GradScaler()
        # self.criterion = nn.CrossEntropyLoss()
        # pos_count = sum([(data.y == 1).sum().item() for data in train_loader.dataset])
        # neg_count = sum([(data.y == 0).sum().item() for data in train_loader.dataset])
        # pos_weight = neg_count / pos_count
        # class_weights = torch.tensor([1.0, pos_weight], device=device)  # 0类权重1，1类权重pos_weight
        # self.criterion = nn.CrossEntropyLoss(weight=class_weights)  # 加权损失
        # pos_weight = torch.tensor([5.0], device=device)  # 正样本权重 5 倍（最稳定）
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0], device=device))

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for data in tqdm(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            with autocast():
                logits = self.model(data)
                loss = self.criterion(logits, data.y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
        self.scheduler.step()
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        yt, yp, ypb = [], [], []
        threshold = 0.5
        for data in loader:
            data = data.to(self.device)
            logits = self.model(data)
            pred = logits.argmax(1)
            prob = torch.softmax(logits, dim=1)[:, 1]  # 正类概率
            pred = (prob > threshold).long()
            yt.extend(data.y.cpu().numpy())
            yp.extend(pred.cpu().numpy())
            ypb.extend(prob.cpu().numpy())
        return compute_metrics(np.array(yt), np.array(yp), np.array(ypb))

    def run(self, epochs=50):
        best_mcc = -1
        for e in range(epochs):
            loss = self.train_epoch()
            val = self.evaluate(self.val_loader)
            print(f"Epoch {e+1:02d} | Loss {loss:.4f} | Val MCC {val['mcc']:.4f} | F1 {val['f1']:.4f}")
            if val['mcc'] > best_mcc:
                best_mcc = val['mcc']
                torch.save(self.model.state_dict(), "best_egnn_node.pt")

        self.model.load_state_dict(torch.load("best_egnn_node.pt"))
        test = self.evaluate(self.test_loader)
        print("\n==================== 测试结果 ====================")
        print(f"  ROC_AUC:   {test['roc_auc']:.4f}")
        print(f"  PR_AUC:    {test['pr_auc']:.4f}")
        print(f"  MCC:       {test['mcc']:.4f}")
        print(f"  F1:        {test['f1']:.4f}")
        print(f"  Precision: {test['precision']:.4f}")
        print(f"  Recall:    {test['recall']:.4f}")
        print("==================================================")

# ===================== 主程序 =====================
if __name__ == "__main__":
    data_train = torch.load("get_graph_edge_attr/data/pyg_graph_datas_495_train_edge_attr.pt",weights_only=False)
    data_test = torch.load("get_graph_edge_attr/data/pyg_graph_datas_117_train_edge_attr.pt",weights_only=False)

    np.random.seed(42)
    idx = np.random.permutation(len(data_train))
    train_idx, val_idx = idx[:int(0.8*len(idx))], idx[int(0.8*len(idx)):]

    train_data = [data_train[i] for i in train_idx]
    val_data = [data_train[i] for i in val_idx]
    test_data = data_test

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EGNNClassifier(
        in_node_nf=train_data[0].x.shape[1],
        hidden_nf=64,
        num_classes=2,
        in_edge_nf=5
    )

    trainer = Trainer(model, train_loader, val_loader, test_loader, device)
    trainer.run(epochs=50)
