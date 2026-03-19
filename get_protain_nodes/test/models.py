import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.dim() == 3:
            batch_size, seq_len, d_model = x.shape
            pe = self.pe[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
            x = x + pe
        else:
            x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class GraphContextExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.node_feature_dim = config.node_feature_dim
        self.context_dim = config.context_dim
        self.d_model = config.d_model

        try:
            from torch_geometric.nn import GPSConv, EGConv
            from torch.nn import Linear as Lin
            self.use_gps = True
            
            self.node_emb = Lin(config.node_feature_dim, config.d_model)
            
            egconv = EGConv(config.d_model, config.d_model)
            self.gps_conv = GPSConv(config.d_model, egconv, heads=4, dropout=config.dropout)
            
            self.stat_encoder = nn.Sequential(
                nn.Linear(6, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, 128),
                nn.LayerNorm(128),
                nn.GELU()
            )
            
            self.fusion = nn.Sequential(
                nn.Linear(config.d_model + 128, config.context_dim),
                nn.LayerNorm(config.context_dim),
                nn.GELU(),
                nn.Dropout(config.dropout)
            )
        except ImportError:
            self.use_gps = False

        if not self.use_gps:
            self.encoder = nn.Sequential(
                nn.Linear(config.node_feature_dim, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.GELU(),
            )

            self.pool = nn.AdaptiveAvgPool1d(1)

            self.stat_encoder = nn.Sequential(
                nn.Linear(6, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, 128),
                nn.LayerNorm(128),
                nn.GELU()
            )

            self.fusion = nn.Sequential(
                nn.Linear(config.d_model + 128, config.context_dim),
                nn.LayerNorm(config.context_dim),
                nn.GELU(),
                nn.Dropout(config.dropout)
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def compute_stats(self, graph):
        node_features = graph.x
        stats = []
        stats.append(node_features.mean().item())
        stats.append(node_features.std().item())
        stats.append(node_features.max().item())
        stats.append(node_features.min().item())
        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
            num_edges = graph.edge_index.size(1)
            avg_degree = 2 * num_edges / graph.num_nodes if graph.num_nodes > 0 else 0
        else:
            avg_degree = 0
        stats.append(avg_degree)
        stats.append(graph.num_nodes)
        return torch.tensor(stats, dtype=torch.float).unsqueeze(0)

    def forward(self, protein_graph):
        if self.use_gps:
            x = self.node_emb(protein_graph.x)
            edge_index = protein_graph.edge_index
            
            x = self.gps_conv(x, edge_index)
            
            pooled = x.mean(dim=0).unsqueeze(0)
            stats = self.compute_stats(protein_graph).to(x.device)
            stats_encoded = self.stat_encoder(stats)
            combined = torch.cat([pooled, stats_encoded], dim=-1)
            context = self.fusion(combined)
        else:
            node_features = protein_graph.x
            encoded = self.encoder(node_features)
            pooled = encoded.mean(dim=0).unsqueeze(0)
            stats = self.compute_stats(protein_graph).to(node_features.device)
            stats_encoded = self.stat_encoder(stats)
            combined = torch.cat([pooled, stats_encoded], dim=-1)
            context = self.fusion(combined)
        
        return context


class TransformerProteinGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.node_feature_dim = config.node_feature_dim
        self.latent_dim = config.latent_dim
        self.n_classes = config.n_classes
        self.context_dim = config.context_dim
        self.d_model = config.d_model

        self.noise_proj = nn.Linear(config.latent_dim, config.d_model)
        self.class_proj = nn.Linear(config.n_classes, config.d_model)
        self.context_proj = nn.Linear(config.context_dim, config.d_model)

        self.node_prompts = nn.Parameter(torch.randn(1, 50, config.d_model) * 0.02)

        self.position_encoding = PositionalEncoding(config.d_model, config.dropout, max_len=2000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.node_feature_dim)
        )

        self.output_norm = nn.LayerNorm(config.node_feature_dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0, std=0.02)
            elif p.dim() == 1:
                nn.init.zeros_(p)

    def forward(self, noise, class_cond, context, n_generate=1):
        batch_size = noise.size(0)

        noise_feat = self.noise_proj(noise).unsqueeze(1)
        class_feat = self.class_proj(class_cond).unsqueeze(1)
        context_feat = self.context_proj(context).unsqueeze(1)

        cond_feat = noise_feat + class_feat + context_feat

        if n_generate > 1:
            prompts = self.node_prompts[:, :n_generate, :].expand(batch_size, -1, -1)
            seq_input = prompts + cond_feat
        else:
            seq_input = cond_feat

        seq_input = self.position_encoding(seq_input)

        transformer_output = self.transformer_encoder(seq_input)

        raw_output = self.output_proj(transformer_output)

        output = self.output_norm(raw_output)

        return output


class TransformerProteinDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.node_feature_dim = config.node_feature_dim
        self.d_model = config.disc_d_model

        self.input_proj = nn.Sequential(
            nn.Linear(config.node_feature_dim, config.disc_d_model),
            nn.LayerNorm(config.disc_d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        self.position_encoding = PositionalEncoding(config.disc_d_model, config.dropout, max_len=2000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.disc_d_model,
            nhead=config.disc_nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.disc_num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(config.disc_d_model, config.disc_d_model // 2),
            nn.LayerNorm(config.disc_d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.disc_d_model // 2, config.disc_d_model // 4),
            nn.LayerNorm(config.disc_d_model // 4),
            nn.GELU(),
            nn.Linear(config.disc_d_model // 4, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0, std=0.02)

    def forward(self, node_features):
        batch_size = node_features.size(0)
        x = self.input_proj(node_features)
        x = self.position_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        output = self.classifier(x)
        return output


class SyntheticNodeConnector:
    def __init__(self, edge_predictor=None, threshold=0.5, max_edges_ratio=0.3):
        self.edge_predictor = edge_predictor
        self.threshold = threshold
        self.max_edges_ratio = max_edges_ratio

    def determine_connections(self, original_features, synthetic_features):
        """
        使用边预测器连接节点
        """
        n_original = original_features.size(0)
        n_synthetic = synthetic_features.size(0)
        device = original_features.device

        if n_original == 0:
            return torch.zeros(2, 0, dtype=torch.long, device=device)

        if n_synthetic == 0:
            return torch.zeros(2, 0, dtype=torch.long, device=device)

        edges_list = []

        all_features = torch.cat([original_features, synthetic_features], dim=0)
        
        with torch.no_grad():
            batch_size = 256
            all_probs = []
            
            for i in range(0, n_synthetic, batch_size):
                end_idx = min(i + batch_size, n_synthetic)
                synth_batch = synthetic_features[i:end_idx]
                
                step = min(synth_batch.size(0), n_original)
                
                probs_synth_to_orig = []
                for j in range(0, n_original, step):
                    orig_end = min(j + step, n_original)
                    orig_batch = original_features[j:orig_end]
                    
                    max_len = max(synth_batch.size(0), orig_batch.size(0))
                    synth_padded = F.pad(synth_batch, (0, 0, 0, max_len - synth_batch.size(0)))
                    orig_padded = F.pad(orig_batch, (0, 0, 0, max_len - orig_batch.size(0)))
                    
                    logits = self.edge_predictor(synth_padded, orig_padded)
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    probs = torch.sigmoid(logits)[:synth_batch.size(0), :orig_batch.size(0)]
                    probs_synth_to_orig.append(probs)
                
                probs_synth_to_orig = torch.cat(probs_synth_to_orig, dim=1)
                
                for bi in range(probs_synth_to_orig.size(0)):
                    sims = probs_synth_to_orig[bi]
                    max_edges = max(1, int(n_original * self.max_edges_ratio))
                    k = min(max_edges, n_original)
                    topk_vals, topk_indices = torch.topk(sims, k)
                    
                    valid_mask = topk_vals > self.threshold
                    valid_indices = topk_indices[valid_mask]
                    
                    synth_node_idx = i + bi
                    for idx in valid_indices:
                        edges_list.append([n_original + synth_node_idx, idx.item()])
                        edges_list.append([idx.item(), n_original + synth_node_idx])

            if n_synthetic > 1:
                for i in range(0, n_synthetic, batch_size):
                    end_idx = min(i + batch_size, n_synthetic)
                    synth_batch = synthetic_features[i:end_idx]
                    
                    if synth_batch.size(0) > 1 and synthetic_features.size(0) > 1:
                        max_len = max(synth_batch.size(0), synthetic_features.size(0))
                        synth_padded = F.pad(synth_batch, (0, 0, 0, max_len - synth_batch.size(0)))
                        synth_full_padded = F.pad(synthetic_features, (0, 0, 0, max_len - synthetic_features.size(0)))
                        
                        logits = self.edge_predictor(synth_padded, synth_full_padded)
                        if logits.dim() == 1:
                            logits = logits.unsqueeze(0)
                        probs = torch.sigmoid(logits)[:synth_batch.size(0), :synthetic_features.size(0)]
                        if probs.dim() == 1:
                            probs = probs.unsqueeze(0)
                        
                        for bi in range(probs.size(0)):
                            sims = probs[bi].clone()
                            if sims.dim() == 0:
                                continue
                            sims[bi] = -1
                            max_edges = max(1, int(n_synthetic * self.max_edges_ratio))
                            k = min(max_edges, n_synthetic - 1)
                            topk_vals, topk_indices = torch.topk(sims, k)
                            
                            valid_mask = topk_vals > self.threshold
                            valid_indices = topk_indices[valid_mask]
                            
                            synth_node_idx = i + bi
                            for idx in valid_indices:
                                idx = idx.item()
                                if idx != synth_node_idx:
                                    edges_list.append([n_original + synth_node_idx, n_original + idx])
                                    edges_list.append([n_original + idx, n_original + synth_node_idx])

        if edges_list:
            edges = torch.tensor(edges_list, dtype=torch.long).T
            edges = edges.unique(dim=1)
        else:
            edges = torch.zeros(2, 0, dtype=torch.long, device=device)

        return edges