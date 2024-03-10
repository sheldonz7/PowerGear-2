import imp
from typing import Union, Tuple
import numpy as np
import pandas as pd
import __init__
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear, BatchNorm1d,LSTM 
from torch_sparse import SparseTensor, matmul, masked_select_nnz
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size, OptPairTensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
import __init__
from utils.base_func import masked_edge_index,masked_edge_attr


class HECGATConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int,int]], out_channels: int, dim=1, num_relations=1,
                 num_heads=1, concat=True, dropout=0.6, bias=True, **kwargs):
        super(HECGATConv, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.dropout = dropout

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        
        self.relation_weight = nn.ModuleList()
        self.relation_att = nn.ModuleList()
        
        for i in range(num_relations):
            self.relation_weight.append(nn.Linear(in_channels[0], out_channels * num_heads, bias=False))
            self.relation_att.append(nn.Linear(2 * out_channels, num_heads, bias=False))
        
        self.lin_r = nn.Linear(in_channels[1], out_channels * num_heads, bias=False)
        self.attr_fc = nn.Linear(dim, in_channels[0], bias=False)
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        self.dim = dim

    def reset_parameters(self):
        for weight in self.relation_weight:
            weight.reset_parameters()
        for att in self.relation_att:
            att.reset_parameters()
        self.lin_r.reset_parameters()
        self.attr_fc.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_weight=None, edge_type=None, size=None):
        if isinstance(x, torch.Tensor):
            x = (x, x)
        
        out = torch.zeros(x[0].size(0), self.num_heads, self.out_channels, device=x[0].device)
        
        if self.dim > 1:
            edge_weight = self.attr_fc(edge_weight)
        
        # iterate through each relation type, with separate relation weight and attention         
        for i in range(self.num_relation):
            edge_index_i = edge_index[:, edge_type == i]
            edge_weight_i = edge_weight[edge_type == i] if edge_weight is not None else None
            
            x_i = self.propagate(edge_index_i, x=x, edge_weight=edge_weight_i, relation_idx=i)
            out += x_i.view(-1, self.num_heads, self.out_channels)
        
        if x[1] is not None:
            out += self.lin_r(x[1]).view(-1, self.num_heads, self.out_channels)

        if self.concat:
            out = out.view(-1, self.num_heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        if self.bias is not None:
            out += self.bias
        
        return out
    
    def message(self, edge_index_i, x_i, x_j, edge_weight_i, relation_idx):
        x_j = self.relation_weight[relation_idx](x_j).view(-1, self.num_heads, self.out_channels)
        if edge_weight_i is not None:
            edge_weight_i = edge_weight_i.view(-1, 1, 1)
            x_j = edge_weight_i * x_j
        
        alpha = torch.cat([x_i, x_j], dim=-1)
        alpha = self.relation_att[relation_idx](alpha)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, edge_index_i[0])
        
        return x_j * alpha.view(-1, self.num_heads, 1)
    
    def update(self, aggr_out):
        return aggr_out
    
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels,
                                             self.out_channels, self.num_heads)