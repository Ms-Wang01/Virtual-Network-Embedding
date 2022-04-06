from scheduler.base_scheduler import BaseScheduler
import torch

class WorstFitScheduler(BaseScheduler):
    def get_action(self, obs):
        virt_graph, subs_graph = obs

        max_diff = -torch.inf
        max_action = None

        for i, j in self.schedulable_pair(virt_graph, subs_graph):
            virt_feat = virt_graph.x[i][:self.n_res]
            subs_feat = subs_graph.x[j][:self.n_res]

            diff = torch.sum(subs_feat - virt_feat)
            
            if max_diff < diff:
                max_diff = diff
                max_action = i, j

        return max_action