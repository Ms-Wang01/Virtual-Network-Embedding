import torch
from scheduler.base_scheduler import BaseScheduler


class FirstFitScheduler(BaseScheduler):
    def get_action(self, obs):
        virt_graph, subs_graph = obs
        
        for i, j in self.schedulable_pair(virt_graph, subs_graph):
            return i, j