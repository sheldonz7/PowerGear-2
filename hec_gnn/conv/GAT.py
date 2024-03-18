import imp
from typing import Union, Tuple, Optional
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
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import __init__
from utils.base_func import masked_edge_index,masked_edge_attr
import pdb

class HECGATConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int,int]],
        out_channels: int,
        num_heads: int = 1,
        dim ,
        edge_dim: Optional[int] = None,
        negative_slope: float = 0.2,
        num_relations: int = 1,
        add_self_loops: bool = True,
        concat: bool = True,
        dropout: float = 0.6,
        bias: bool = True,
        fill_value: Union[float, Tensor, str] = 'mean',
        share_weights: bool = False,
        **kwargs
    ):
        
        
        super(HECGATConv, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat = concat
        self.negative_slope = negative_slope
        self.edge_dim = edge_dim
        self.add_self_loops = add_self_loops
        self.fill_value = fill_value
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        
        self.relation = nn.ModuleList()
        

        
        self.att_weights = nn.ParameterList()

        for i in range(num_relations):
            self.relation_weight.append(nn.Linear(in_channels[0], out_channels * num_heads, bias=False))
            self.att_weights.append(nn.Parameter(torch.Tensor(1, num_heads, out_channels)))
        
        self.att = Parameter(torch.empty(1, heads, out_channels))

        # linear layers for source and target node features transformation
        self.lin_l = nn.Linear(in_channels[0], out_channels * num_heads, bias=False)
        # self.lin_r = nn.Linear(in_channels[0], out_channels * num_heads, bias=False)
        self.lin_r = self.lin_l
        
        if edge_dim is not None
        self.lin_edge = nn.Linear(in_channels[1], out_channels * num_heads, bias=False)

        # used to map edge features into the same dimension as node features
        self.attr_fc = nn.Linear(dim, in_channels[1], bias=False)
        



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
    
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, edge_type: OptTensor = None,size: Size = None):
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        out = torch.zeros(x[0].size(0), self.num_heads, self.out_channels, device=x[0].device)
        
        if self.dim > 1:
            edge_weight = self.attr_fc(edge_weight)
        
        pdb.set_trace()
        # iterate through each relation type, with separate relation weight and attention         
        for i in range(self.num_relations):
            tmp = masked_edge_index(edge_index, edge_type == i)
            
            tmp_out = self.propagate(tmp, x=x, edge_weight=masked_edge_attr(edge_weight, edge_type == i), size=size, relation_idx=i)
            out += tmp_out.view(-1, self.num_heads, self.out_channels)
        
        if x[1] is not None:
            out += self.lin_r(x[1]).view(-1, self.num_heads, self.out_channels)

        if self.concat:
            out = out.view(-1, self.num_heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        if self.bias is not None:
            out += self.bias
        
        return out
    
    # x_i is the source/central node feature, x_j is the target node feature
    # 

    def message(self, edge_index, x_i, x_j, edge_weight, relation_idx):
        # x_j = self.relation_weight[relation_idx](x_j).view(-1, self.num_heads, self.out_channels)
        # if edge_weight_i is not None:
        #     edge_weight_i = edge_weight_i.view(-1, 1, 1)
        #     x_j = edge_weight_i * x_j
        
        H,C = self.num_heads, self.out_channels
        
        pdb.set_trace()
        
        
        # apply relational weight (linear layer) to the edge embeddings, as well as target and source node embeddings
        edge_weight_msg = self.relation_weight[relation_idx](msg).view(-1, H, C)
        x_i_msg = self.lin_l(x_i).view(-1, H, C)
        x_j_msg = self.lin_r(x_j).view(-1, H, C)

        # use edge attributes as message
        if edge_weight is not None:
            msg = edge_weight
        else:
            msg = x_j

        

        # still use source and target node features to calculate attention
        alpha = torch.cat([x_i, x_j], dim=-1)
        alpha = self.relation_att[relation_idx](alpha)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, edge_index[0])
        
        # output in the shape of (N, num_heads, out_channels)
        return msg * alpha.view(-1, self.num_heads, 1) 
    
    def update(self, aggr_out):
        return aggr_out
    
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels,
                                             self.out_channels, self.num_heads)