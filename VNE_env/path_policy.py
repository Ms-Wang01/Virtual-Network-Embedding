
def find_path(info, src: int, dst: int, flow: float):
    subs_graph = info.subs_graph

    queue = [[src]]
    visited = [False] * len(subs_graph.nodes)

    while queue:
        path = queue.pop()

        last = path[-1]
        
        if last == dst:
            return path

        visit_list = []
        
        for tgt in subs_graph[last]:
            if not visited[tgt]:
                r_flow = subs_graph.edges[(last, tgt)]['features']
                visit_list.append((r_flow, tgt))

        visit_list.sort(reverse=True, key=lambda x: x[0])
        
        for r_flow, tgt in visit_list:
            if flow <= r_flow:
                visited[tgt] = True
                new_path = path[:]
                new_path.append(tgt)
                queue.append(new_path)
    
    return None
