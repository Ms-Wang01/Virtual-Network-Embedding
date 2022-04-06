from typing import Tuple
import torch
from torch_geometric.data import Data
from argparse import Namespace

class BaseScheduler(object):
    def __init__(self, args: Namespace):
        self.args = args
        self.n_res = args.node_num_res
    
    def schedulable_pair(self, virt_graph: Data, subs_graph: Data) -> Tuple[int, int]:
        for i, virt_feat in enumerate(virt_graph.x):
            if not virt_graph.mask[i]:
                continue

            for j, subs_feat in enumerate(subs_graph.x):
                if self.args.no_same_place and not subs_graph.mask[j]:
                    continue

                if torch.all(virt_feat[:self.n_res] <= subs_feat[:self.n_res]):
                    yield i, j


    def reset(self) -> None:
        pass

    def get_action(self, obs: Tuple[Data, Data]) -> Tuple[int, int]:
        raise NotImplementedError