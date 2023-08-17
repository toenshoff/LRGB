from torch_geometric.graphgym.register import register_config


@register_config('custom_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """
    # Use residual connections between the GNN layers.
    cfg.gnn.residual = False
    cfg.gnn.heads = 4
    cfg.gnn.attn_dropout = 0.1

    cfg.gnn.use_vn = True
    cfg.gnn.vn_pooling = 'add'

    cfg.gnn.norm_type = 'layer'
