from typing import Tuple
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from argparse import Namespace
from scheduler.base_scheduler import BaseScheduler
from collections import defaultdict
from random import sample

class GCNModule(nn.Module):
    def __init__(self, input_dim:int, k: int, hid_dim: int) -> None:
        super(GCNModule, self).__init__()

        self.conv = nn.ModuleList([
            GCNConv(input_dim if i == 0 else hid_dim, hid_dim)
            for i in range(k)    
        ])


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.conv:
            x = conv(x, edge_index)
            x = F.relu(x)

        return x

class GCNScheduler(BaseScheduler):
    def __init__(self, args: Namespace, k: int, hid_dim: int, sg_nodes: int, get_prob: bool=False):
        super().__init__(args)
        self.virt_idx = None
        self.virt_gcn = GCNModule(4, k, hid_dim)
        self.subs_gcn = GCNModule(6, k, hid_dim)
        self.lin = nn.Linear(hid_dim * 2 * sg_nodes, sg_nodes)
        self.get_prob = get_prob

    def virt_graph_handler(self, virt_graph: Data):
        if self.virt_idx != virt_graph.idx:
            self.virt_idx = virt_graph.idx
            self.virt_emb = self.virt_gcn(virt_graph)

    def train(self) -> None:
        self.virt_gcn.train()
        self.subs_gcn.train()
        self.lin.train()

    def eval(self) -> None:
        self.virt_gcn.eval()
        self.subs_gcn.eval()
        self.lin.eval()

    def convert_graph(self, graph: Data):
        edge_index = graph.edge_index[0]
        res_list = []

        for i in range(graph.edge_attr.shape[1]):
            res = torch.zeros(graph.x.shape[0], dtype=torch.float)
            res.scatter_add_(0, edge_index, graph.edge_attr.T[i])
            res = res.reshape(-1, 1)
            res_list.append(res)
        
        res = torch.cat(res_list, dim=1)
        
        x = torch.cat([graph.x, res], dim=1)
        x = F.normalize(x, dim=0)
        graph.x = x

        return graph

    def get_action(self, obs: Tuple[Data, Data]):
        assign_dict = defaultdict(list)
        for i, j in self.schedulable_pair(*obs):
            assign_dict[i].append(j)

        virt_graph, subs_graph = map(self.convert_graph, obs)

        self.virt_graph_handler(virt_graph)

        virt_node = sample(assign_dict.keys(), 1)[0]

        virt_emb = self.virt_emb[virt_node]
        subs_emb = self.subs_gcn(subs_graph)

        virt_emb = virt_emb.repeat(subs_emb.shape[0], 1)

        emb = torch.flatten(torch.cat((virt_emb, subs_emb)))
        emb = F.softmax(self.lin(emb), dim=0)

        candidate_subs_node = torch.tensor(assign_dict[virt_node]).to(emb.device)
        subs_node = torch.multinomial(emb[candidate_subs_node], 1)[0]

        prob = emb[candidate_subs_node[subs_node]]
        subs_node = candidate_subs_node[subs_node].item()

        if self.get_prob:
            return prob, (virt_node, subs_node)
        else:
            return virt_node, subs_node
