import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer with edge features
    
"""
from layers.graph_transformer_edge_layer import GraphTransformerLayer
# from layers.graph_transformer_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        # num_atom_type = net_params['num_atom_type']
        # num_bond_type = net_params['num_bond_type']

        # in_dim_node = net_params['in_dim']

        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        max_wl_role_index = 37 # this is maximum graph size in the dataset
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)

        # What should replace num_atom_type here?
        # self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)
        # torch.unique(dataset.train[0][0].ndata['feat'],dim=0).size(0)
        # self.embedding_h = nn.Embedding(10000, 256)
        # self.embedding_h = nn.Embedding(100, hidden_dim)
        # self.embedding_h = nn.Embedding(in_dim_node, hidden_dim)
        # print("arch embedding_h = ", self.embedding_h)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        # self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem        
        
    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):

        # input embedding
        # h = torch.tensor(h).to(0).long() # device: 0 is cuda gpu
        # h = h.clone().detach().long()
        # h = h.to(torch.long)

        # print()
        # print("(1) forward embedding_h, h = ", self.embedding_h, h.size())
        # print(h)
        # h = self.embedding_h(h)
        # print("(2) forward embedding_h, h = ", self.embedding_h, h.size())
        # print(h)
        # print()
        
        h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0),1).to(self.device)
        e = self.embedding_e(e)   
        
        # print(h.shape, e.shape)
        
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
            # h = conv(g, h)
        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        # return self.MLP_layer(hg)
        return hg
        
    # def loss(self, scores, targets):
    #     # loss = nn.MSELoss()(scores,targets)
    #     loss = nn.L1Loss()(scores, targets)
    #     return loss
