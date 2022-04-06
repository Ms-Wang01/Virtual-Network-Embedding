import networkx as nx
import numpy as np

def shortest_job_first(env) -> nx.DiGraph:
    shortest_job = None
    shortest_runtime = np.inf

    for virt_graph in env.virt_graphs:
        if virt_graph.graph['scheduled']:
            continue
        
        find = False
        for subs_node in env.subs_graph.nodes:
            subs_feat = env.subs_graph.nodes[subs_node]['features']
            for virt_node in virt_graph.nodes:
                virt_feat = virt_graph.nodes[virt_node]['features']
                if np.all(virt_feat < subs_feat):
                    if virt_graph.graph['runtime'] < shortest_runtime:
                        shortest_runtime = virt_graph.graph['runtime']
                        shortest_job = virt_graph
                        find = True          
                        break                  
            if find:
                break

    return shortest_job