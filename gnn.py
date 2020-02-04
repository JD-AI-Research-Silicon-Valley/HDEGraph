import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, math

class ginLayer(nn.Module):
    def __init__(self, input_dim, proj_dim=512, dropout=0.1, num_hop=3, batch_norm=False, num_rel=3):
        super(ginLayer, self).__init__()

        self.proj_dim = proj_dim
        self.num_hop = num_hop
        self.num_rel = num_rel
        self.epsilon = nn.Parameter(torch.Tensor([0])) # intialized with 0

        self.fa = nn.Sequential(nn.Linear(input_dim + proj_dim, proj_dim), nn.Dropout(dropout))

        for i in range(self.num_hop):
            for j in range(self.num_rel):
                setattr(self, "mlp{}{}".format(i+1, j+1), nn.Sequential(nn.Linear(input_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout)))

    def forward(self, input, input_mask, adj):
        # input: bs x max_nodes x node_dim
        # input_mask: bs x max_nodes
        # adj: bs x num_rel x max_nodes x max_nodes

        cur_input = input.clone()

        for i in range(self.num_hop):

            # replicate input
            multi_input = torch.stack([cur_input for i in range(self.num_rel)], dim=1) # bs x num_rel x max_nodes x node_dim

            # integrate neighbor information
            nb_output = torch.matmul(adj, multi_input) * input_mask.unsqueeze(-1).unsqueeze(1)  # bs x num_rel x max_nodes x node_dim

            # add cur node
            cur_update = (1 + self.epsilon)*multi_input + nb_output

            # apply different mlps for different relations
            update = torch.mean(torch.stack([getattr(self, "mlp{}{}".format(i+1, j+1))(cur_update[:,j,:,:].squeeze(1)) \
                                for j in range(self.num_rel)], dim=1), dim=1)* input_mask.unsqueeze(-1) # bs x max_nodes x node_dim

            # get gate values
            gate = torch.sigmoid(self.fa(torch.cat((update, cur_input), -1))) * input_mask.unsqueeze(
                -1)  # bs x max_node x node_dim

            # apply gate values
            cur_input = gate * update + (1 - gate) * cur_input  # bs x max_node x node_dim

        return cur_input

class gatLayer(nn.Module):
    def __init__(self, input_dim, proj_dim=512, dropout=0.1, num_hop=3, num_rel=2, batch_norm=False):
        super(gatLayer, self).__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.num_hop = num_hop
        self.gcn_num_rel = num_rel

        self.sfm = nn.Softmax(-1)

        for i in range(self.gcn_num_rel):
            setattr(self, "fr{}".format(i+1), nn.Sequential(nn.Linear(input_dim, proj_dim), nn.Dropout(dropout)))

            # attention weight
            setattr(self, "fa{}".format(i+1), nn.Linear(input_dim, input_dim, bias=False))

        self.fs = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.Dropout(dropout))

        self.fg = nn.Sequential(nn.Linear(input_dim + proj_dim, proj_dim), nn.Dropout(dropout))

    def forward(self, input, input_mask, adj):
        # input: bs x max_nodes x node_dim
        # input_mask: bs x max_nodes
        # adj: bs x 3 x max_nodes x max_nodes
        # num_layer: number of layers; note that the parameters of all layers are shared

        cur_input = input.clone()

        for i in range(self.num_hop):

            # get attention
            att_list = []
            for j in range(self.gcn_num_rel):
                att = torch.bmm(getattr(self, "fa{}".format(j+1))(cur_input), 
                                cur_input.transpose(1,2).contiguous()) /math.sqrt(self.input_dim) # bs x max_nodes x max_nodes
                zero_vec = -9e15*torch.ones_like(att)
                att = torch.where(adj[:,j,:,:].squeeze(1) > 0, att, zero_vec)
                att_list.append(self.sfm(att))

            att_matrices = torch.stack(att_list, dim=1)
            
            # integrate neighbor information
            nb_output = torch.stack([getattr(self, "fr{}".format(i+1))(cur_input) for i in range(self.gcn_num_rel)],
                                    1) * input_mask.unsqueeze(-1).unsqueeze(1)  # bs x 2 x max_nodes x node_dim

            # apply different types of connections, which are encoded in adj matrix
            update = torch.sum(torch.matmul(att_matrices, torch.matmul(adj,nb_output)), dim=1, keepdim=False) + \
                     self.fs(cur_input) * input_mask.unsqueeze(-1)  # bs x max_node x node_dim

            # get gate values
            gate = torch.sigmoid(self.fg(torch.cat((update, cur_input), -1))) * input_mask.unsqueeze(
                -1)  # bs x max_node x node_dim

            # apply gate values
            cur_input = gate * torch.tanh(update) + (1 - gate) * cur_input  # bs x max_node x node_dim

        return cur_input