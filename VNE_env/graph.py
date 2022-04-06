import numpy as np
import networkx as nx
from typing import NamedTuple
from argparse import Namespace
from numpy.random import RandomState

class NodeType(object):
    NODE = 0
    AGG = 1
    CORE = 2

class GraphInfo(NamedTuple):
    n_feat_low: int
    n_feat_high: int
    n_res: int
    e_feat_low: int
    e_feat_high: int
    e_res: int
    seed: int
    rng: RandomState

def gen_node_features(n: int, gi: GraphInfo) -> np.ndarray:
    features = gi.rng.randint(
        low=gi.n_feat_low,
        high=gi.n_feat_high,
        size=(n, gi.n_res)
    ).astype(np.float64)

    return features

def gen_empty_node_features(n: int, gi: GraphInfo) -> np.ndarray:
    features = np.zeros((n, gi.n_res))
    return features

def gen_edge_features(e: int, gi: GraphInfo) -> np.ndarray:
    edge_features = gi.rng.randint(
        low=gi.e_feat_low,
        high=gi.e_feat_high,
        size=(e, gi.e_res)
    ).astype(np.float64)
    return edge_features

def generate_substrate_graph(args: Namespace, rng: RandomState) -> np.ndarray:
    gi = GraphInfo(
        n_feat_low=args.sn_feat_low,
        n_feat_high=args.sn_feat_high,
        n_res=args.node_num_res,
        e_feat_low=args.sl_feat_low,
        e_feat_high=args.sl_feat_high,
        e_res=args.link_num_res,
        seed=args.seed,
        rng=rng
    )
    
    if args.sn_fat_tree:
        return fat_tree(args.pods, gi)
    elif args.sn_erdos_renyi:
        return erdos_renyi(args.num_sn, args.sl_prob, gi)
    elif args.sn_barabasi_albert:
        return erdos_renyi(args.num_sn, args.num_sn_m, gi)
    else:
        raise NotImplementedError

def generate_virtual_graph(args: Namespace, rng: Namespace, pad: int) -> np.ndarray:
    n = rng.randint(args.num_vn_low, args.num_vn_high)

    gi = GraphInfo(
        n_feat_low=args.vn_feat_low,
        n_feat_high=args.vn_feat_high,
        n_res=args.node_num_res,
        e_feat_low=args.vl_feat_low,
        e_feat_high=args.vl_feat_high,
        e_res=args.link_num_res,
        seed=args.seed + pad + 1,
        rng=rng
    )
    
    if args.vn_erdos_renyi:
        return erdos_renyi(n, args.vl_prob, gi)
    elif args.vn_barabasi_albert:
        return erdos_renyi(n, args.num_vn_m, gi)
    else:
        raise NotImplementedError

def fat_tree(pods: int, gi: GraphInfo) -> nx.DiGraph:
    num_hosts = (pods ** 3) // 4
    num_aggs = pods ** 2
    num_cores = num_aggs // 4

    host_feats = gen_node_features(num_hosts, gi)
    core_feats = gen_empty_node_features(num_cores, gi)
    agg_feats = gen_empty_node_features(num_aggs, gi)

    node_feats = np.concatenate([
        host_feats, 
        core_feats, 
        agg_feats
    ], axis=0)

    edge_type = {
        (NodeType.NODE, NodeType.AGG): 0,
        (NodeType.AGG, NodeType.NODE): 1,
        (NodeType.CORE, NodeType.AGG): 2,
        (NodeType.AGG, NodeType.CORE): 3,
        (NodeType.AGG, NodeType.AGG): 4,
    }

    hosts = [
        (i, {'features': node_feats[i], 'type': NodeType.NODE})
        for i in range(num_hosts)
    ]

    cores = [
        (i, {'features': node_feats[i], 'type': NodeType.CORE})
        for i in range(num_hosts, num_hosts + num_cores)
    ]

    aggs = [
        (i, {'features': node_feats[i], 'type': NodeType.AGG})
        for i in range(num_hosts + num_cores,
                    num_hosts + num_cores + num_aggs)
    ]

    g = nx.DiGraph()
    g.add_nodes_from(hosts)
    g.add_nodes_from(cores)
    g.add_nodes_from(aggs)
    host_offset = 0

    edge_idx = 0
    for pod in range(pods):
        core_offset = 0
        for sw in range(pods // 2):
            switch = aggs[(pod * pods) + sw][0]
            for port in range(pods // 2):
                core_switch = cores[core_offset][0]
                features = np.array([gi.e_feat_high] * gi.e_res).astype(np.float64)
                g.add_edge(switch, core_switch, features=features, type=4, idx=edge_idx)
                edge_idx += 1
                g.add_edge(core_switch, switch, features=features, type=3, idx=edge_idx)
                edge_idx += 1
                core_offset += 1
    
            for port in range(pods // 2, pods):
                lower_switch = aggs[(pod*pods) + port][0]
                features = np.array([gi.e_feat_high * 2] * gi.e_res).astype(np.float64)
                g.add_edge(switch, lower_switch, features=features, type=2, idx=edge_idx)
                edge_idx += 1
                g.add_edge(lower_switch, switch, features=features, type=2, idx=edge_idx)
                edge_idx += 1

        for sw in range(pods // 2, pods):
            switch = aggs[(pod * pods) + sw][0]
            for port in range(pods // 2, pods):
                host = hosts[host_offset][0]
                features = np.array([gi.e_feat_high * 4] * gi.e_res).astype(np.float64)
                g.add_edge(switch, host, features=features, type=1, idx=edge_idx)
                edge_idx += 1
                g.add_edge(host, switch, features=features, type=0, idx=edge_idx)
                edge_idx += 1
                host_offset += 1

    return g

def erdos_renyi(n: int, p: float, gi: GraphInfo) -> nx.DiGraph:
    r = nx.fast_gnp_random_graph(n, p, seed=gi.seed)

    num_nodes = len(r.nodes)
    num_edges = len(r.edges)

    node_feats = gen_node_features(num_nodes, gi)
    edge_feats = gen_edge_features(num_edges, gi)
    
    g = nx.DiGraph()

    for node in r.nodes:
        g.add_node(
            node, 
            features=node_feats[node],
            type=NodeType.NODE    
        )

    edge_idx = 0
    for i, (src, dst) in enumerate(r.edges):
        g.add_edge(src, dst, features=edge_feats[i], type=0, idx=edge_idx)
        edge_idx += 1
        g.add_edge(dst, src, features=edge_feats[i], type=0, idx=edge_idx)
        edge_idx += 1

    return g

def barabasi_albert(n: int, m: int, gi: GraphInfo) -> nx.DiGraph:
    r = nx.barabasi_albert_graph(n, m, seed=gi.seed)

    num_nodes = len(r.nodes)
    num_edges = len(r.edges)

    node_feats = gen_node_features(num_nodes, gi)
    edge_feats = gen_edge_features(num_edges, gi)
    
    g = nx.DiGraph()

    for node in r.nodes:
        g.add_node(
            node, 
            features=node_feats[node],
            type=NodeType.NODE    
        )

    edge_idx = 0
    for i, (src, dst) in enumerate(r.edges):
        g.add_edge(src, dst, features=edge_feats[i], type=0, idx=edge_idx)
        edge_idx += 1
        g.add_edge(dst, src, features=edge_feats[i], type=0, idx=edge_idx)
        edge_idx += 1

    return g