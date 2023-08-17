import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_edge_encoder,
act_dict)


@register_edge_encoder('RWSEEdge')
class RWSEEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        pe_dim = len(cfg.posenc_RWSEEdge.kernel.times) + 1
        self.pe_dim = pe_dim
        self.emb_dim = emb_dim

        self.global_edge_dropout = cfg.posenc_RWSEEdge.global_edge_dropout

        self.pe_encoder = nn.Sequential(
            nn.BatchNorm1d(pe_dim),
            nn.Linear(pe_dim, emb_dim),
            act_dict[cfg.gnn.act](),
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )

    def forward(self, batch):
        pe_enc = torch.cat([batch.pestat_RWSEEdge, batch.pestat_RWSESelf], dim=0)

        self_loops = torch.arange(batch.num_nodes, device=pe_enc.device).view(1, -1).tile(2, 1)
        edge_index = torch.cat([batch.edge_index, self_loops], dim=1)

        if 'pestat_RWSEGlobal' in batch:
            global_enc = batch.pestat_RWSEGlobal
            global_edge_index = batch.global_edge_index

            if self.training:
                dropout_mask = torch.rand((global_enc.shape[0],), device=global_enc.device) > self.global_edge_dropout
                global_enc = global_enc[dropout_mask]
                global_edge_index = global_edge_index[:, dropout_mask]

            pe_enc = torch.cat([pe_enc, global_enc], dim=0)
            edge_index = torch.cat([edge_index, global_edge_index], dim=1)

        pe_enc = self.pe_encoder(pe_enc)

        edge_attr = pe_enc
        if batch.edge_attr is not None:
            edge_attr[:batch.num_edges] += batch.edge_attr

        batch.edge_index = edge_index
        batch.edge_attr = edge_attr
        return batch
