import networkx as nx
import numpy as np
from argparse import Namespace
from typing import Tuple
from torch_geometric.data import Data
from random import sample

from scheduler.base_scheduler import BaseScheduler


class GRCScheduler(BaseScheduler):
    def __init__(self, args: Namespace):
        super().__init__(args)
        self.s_vec = None
        self.v_vec = None

    def reset(self) -> None:
        self.s_vec = None
        self.v_vec = None

    def get_action(self, obs: Tuple[Data, Data]) -> Tuple[int, int]:
        virt_graph, subs_graph = obs

        if self.s_vec is None and self.v_vec is None:
            self.s_vec = self.grc_vector(subs_graph, d=0.85)
            self.v_vec = self.grc_vector(virt_graph, d=0.85)

        available_pairs = set(self.schedulable_pair(virt_graph, subs_graph))

        index_s = np.flip(np.argsort(self.s_vec), axis=0)
        index_v = np.flip(np.argsort(self.v_vec), axis=0)

        for i in index_v:
            for j in index_s:
                i, j = map(int, (i, j))
                if (i, j) in available_pairs:
                    return i, j

        return None, None

    def grc_vector(self, g, d):
        net_len = g.x.shape[0]

        x = g.x.cpu().numpy()[:, :self.n_res]
        edge_index = g.edge_index.cpu().numpy()
        edge_attr = g.edge_attr.cpu().numpy()

        bw = np.zeros((x.shape[0], x.shape[0])).astype(np.float32)
        bw[edge_index[0], edge_index[1]] = edge_attr.reshape(-1)

        M = bw.copy()

        for i in range(net_len):
            sum_adj_i = bw[:, i].sum() + 1e-6
            sum_adj_v = np.ones(net_len) * sum_adj_i
            M[:, i] = np.divide(bw[:, i], sum_adj_v)

        c = x / x.sum()

        r0 = c

        Delta = 9999999
        sigma = 0.00001
        while Delta >= sigma:
            r1 = (1-d) * c + d * np.matmul(M, r0)
            tmp = np.linalg.norm(r1 - r0, ord=2)
            Delta = tmp
            r0 = r1

        return r1.reshape(-1)