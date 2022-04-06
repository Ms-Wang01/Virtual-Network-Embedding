from typing import List
import networkx as nx
import numpy as np
from argparse import Namespace
from VNE_env.path_policy import find_path

class Reservation(object):
    def __init__(self, args, virt_graph: nx.DiGraph, subs_graph: nx.DiGraph) -> None:
        self.args = args
        self.virt_tracker = {}
        self.subs_node_tracker = {}
        self.subs_node_feat_tracker = {}
        self.subs_link_tracker = {}
        self.virt_link = set()
        self.virt_graph = virt_graph
        self.subs_graph = subs_graph
        self.find_path = find_path

    def schedulable(self):
        return len(self.virt_tracker) == len(self.virt_graph.nodes)

    def add(self, virt_node, subs_node):
        # reserve node
        self.reserve_node(virt_node, subs_node)
        return self.reserve_link(virt_node, subs_node) # reserve link or not

    def reserve_node(self, virt_node: int, subs_node: int) -> None:
        self.virt_tracker[virt_node] = subs_node
        # reserve node feature
    
        shape = self.virt_graph.nodes[virt_node]['features'].shape[0]

        self.subs_graph.nodes[subs_node]['features'] -= \
            self.virt_graph.nodes[virt_node]['features']

        if subs_node not in self.subs_node_tracker:
            self.subs_node_tracker[subs_node] = set()
        
        if subs_node not in self.subs_node_feat_tracker:
            self.subs_node_feat_tracker[subs_node] = np.zeros((shape,))

        self.subs_node_feat_tracker[subs_node] += \
            self.virt_graph.nodes[virt_node]['features']

        self.subs_node_tracker[subs_node].add(virt_node)

    def can_reserve(self):
        virt_graph = self.virt_graph

        for subs_node in self.subs_graph.nodes:
            if self.args.no_same_place and \
                subs_node in self.subs_node_tracker:
                continue

            subs_feat = self.subs_graph.nodes[subs_node]['features']

            for virt_node in virt_graph.nodes:
                if virt_node in self.virt_tracker:
                    continue
                
                virt_feat = virt_graph.nodes[virt_node]['features']
                if np.all(virt_feat < subs_feat):
                    return True
        
        return False
        
    def reserve_link(self, virt_node: int, subs_node: int) -> bool:
        for adj_virt in self.virt_graph[virt_node]:
            if adj_virt not in self.virt_tracker:
                continue

            if (virt_node, adj_virt) in self.virt_link:
                continue
            
            if (adj_virt, virt_node) in self.virt_link:
                continue

            flow = self.virt_graph.edges[(virt_node, adj_virt)]['features'] 
            adj_subs = self.virt_tracker[adj_virt]

            path = self.find_path(self, subs_node, adj_subs, flow)

            if path is None:
                return False

            for i in range(len(path) - 1):
                src_s, dst_s = path[i], path[i + 1]
                shape = flow.shape[0]

                if (src_s, dst_s) not in self.subs_link_tracker or \
                    (dst_s, src_s) not in self.subs_link_tracker:
                    features = np.zeros((shape,))
                    self.subs_link_tracker[(src_s, dst_s)] = features
                    self.subs_link_tracker[(dst_s, src_s)] = features
                    
                self.subs_graph.edges[(src_s, dst_s)]['features'] -= flow

                self.subs_link_tracker[(src_s, dst_s)] += flow

            self.virt_link.add((adj_virt, virt_node))
            self.virt_link.add((virt_node, adj_virt))
    
        return True