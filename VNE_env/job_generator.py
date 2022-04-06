import numpy as np
from typing import Set, Tuple, List
from VNE_env.graph import generate_virtual_graph
from numpy.random import RandomState
from argparse import Namespace

def generate_virtual_graphs(args: Namespace, rng: RandomState) -> Tuple[Set, List]:
    arrived_virtual_graphs = set()
    not_arrived_virtual_graphs = list()

    num_inits = args.num_vg_init
    num_streams = args.num_vg_stream
    
    num_vg_rts = rng.randint(
        args.num_vg_rt_low,
        args.num_vg_rt_high,
        size=(num_inits + num_streams,)
    ).astype(np.float64)

    vg_intervals = rng.exponential(
        args.vg_interval, 
        size=(num_streams,)
    ).astype(np.float64)

    t = 0

    for virtual_graph_idx in range(num_inits):
        virtual_graph = generate_virtual_graph(args, rng, virtual_graph_idx)

        # initialize virtual graph infomation
        virtual_graph.graph['idx'] = virtual_graph_idx
        virtual_graph.graph['arrive_time'] = t
        virtual_graph.graph['completion_time'] = np.inf
        virtual_graph.graph['scheduled_time'] = np.inf
        virtual_graph.graph['runtime'] = num_vg_rts[virtual_graph_idx]
        virtual_graph.graph['arrived'] = True
        virtual_graph.graph['scheduled'] = False
        virtual_graph.graph['completed'] = False

        arrived_virtual_graphs.add(virtual_graph)

    for virtual_graph_idx in range(num_streams):
        t += vg_intervals[virtual_graph_idx]

        padding_idx = num_inits + virtual_graph_idx

        virtual_graph = generate_virtual_graph(args, rng, padding_idx)

        virtual_graph.graph['idx'] = padding_idx
        virtual_graph.graph['arrive_time'] = t
        virtual_graph.graph['completion_time'] = np.inf
        virtual_graph.graph['scheduled_time'] = np.inf
        virtual_graph.graph['runtime'] = num_vg_rts[padding_idx]
        virtual_graph.graph['arrived'] = False
        virtual_graph.graph['scheduled'] = False
        virtual_graph.graph['completed'] = False

        not_arrived_virtual_graphs.append((t, virtual_graph))

    return arrived_virtual_graphs, not_arrived_virtual_graphs