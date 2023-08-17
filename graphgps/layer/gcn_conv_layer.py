import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym import cfg
import torch_geometric.graphgym.register as register


class GCNConvLayer(nn.Module):
    """Graph Isomorphism Network with Edge features (GINE) layer.
    """
    def __init__(self, dim_in, dim_out, dropout, residual):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual

        self.act = nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            nn.Dropout(self.dropout),
        )
        self.model = pyg_nn.GCNConv(dim_in, dim_out, bias=True)

    def forward(self, batch):
        x_in = batch.x

        batch.x = self.model(batch.x, batch.edge_index)
        batch.x = self.act(batch.x)

        if self.residual:
            batch.x = x_in + batch.x  # residual connection

        return batch
