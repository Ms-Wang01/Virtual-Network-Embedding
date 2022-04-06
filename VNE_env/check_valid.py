import numpy as np
from typing import Tuple
from torch_geometric.data import Data
from VNE_env.reservation_info import Reservation


def check_action(env, virt_node: int, subs_node: int) -> bool:
    if not isinstance(virt_node, int):
        return False

    if not isinstance(subs_node, int):
        return False

    v_feat = env.reservation.virt_graph.nodes[virt_node]['features']
    s_feat = env.subs_graph.nodes[subs_node]['features']

    if env.args.no_same_place and \
        subs_node in env.reservation.subs_node_tracker:
        return False

    return np.all(v_feat <= s_feat)

def check_schedule(_, schedule: Reservation):
    virt_tracker = schedule.virt_tracker
    virt_link = schedule.virt_link
    virt_graph = schedule.virt_graph

    if not virt_graph.graph['arrived']:
        return False

    if virt_graph.graph['scheduled']:
        return False
    
    if virt_graph.graph['completed']:
        return False

    for virt_node in virt_graph.nodes:
        if virt_node not in virt_tracker:
            return False
            
    for src, dst in virt_graph.edges:
        if (src, dst) not in virt_link:
            return False

        if (dst, src) not in virt_link:
            return False
    return True

def check_observe(env, observe: Tuple[Data, Data]):
    virt_graph, subs_graph = observe

    for src, dst in env.subs_graph.edges:
        feat_src_dst = env.subs_graph.edges[(src, dst)]['features']
        feat_dst_src = env.subs_graph.edges[(dst, src)]['features']
        if not np.all(feat_src_dst == feat_dst_src):
            return False

    subs_x = subs_graph.x.numpy()
    subs_edge_attr = subs_graph.edge_attr.numpy()

    virt_x = virt_graph.x.numpy()
    virt_edge_attr = virt_graph.edge_attr.numpy()

    if not (
        np.all(env.sn_low <= subs_x) and 
        np.all(subs_x <= env.sn_high)
    ):
        return False

    if not (
        np.all(env.sl_low <= subs_edge_attr) and 
        np.all(subs_edge_attr <= env.sl_high)
    ):
        return False

    if not (
        np.all(env.vn_low <= virt_x) and 
        np.all(virt_x <= env.vn_high)
    ):
        return False

    if not (
        np.all(env.vl_low <= virt_edge_attr) and 
        np.all(virt_edge_attr <= env.vl_high)
    ):
        return False

    return True

def check_done(env):
    args = env.args
    
    total_virtal_graphs = args.num_vg_init + args.num_vg_stream
    
    if total_virtal_graphs != len(env.finished_virtual_graphs):
        return False

    if len(env.timeline) != 0:
        return False

    subs_node_feats = np.zeros((
        len(env.subs_graph.nodes), 
        env.args.node_num_res
    ))

    subs_link_feats = np.zeros((len(env.subs_graph.edges), env.args.link_num_res))

    for subs_node in env.subs_graph.nodes:
        subs_node_val = env.subs_graph.nodes[subs_node]
        cur_subs_feat = subs_node_val['features']
        subs_node_feats[subs_node] += cur_subs_feat
        
    for subs_link in env.subs_graph.edges:
        subs_link_val = env.subs_graph.edges[subs_link]
        idx = subs_link_val['idx']
        subs_link_feats[idx] = subs_link_val['features']

    if not np.all(subs_node_feats == env.sn_high[:, :args.node_num_res]):
        return False

    if not np.all(subs_link_feats == env.sl_high):
        return False

    return True