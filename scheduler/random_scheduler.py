import torch
from random import sample
from scheduler.base_scheduler import BaseScheduler


class RandScheduler(BaseScheduler):
    def get_action(self, obs):
        virt_graph, subs_graph = obs
        action_list = list(self.schedulable_pair(virt_graph, subs_graph))
        return sample(action_list, 1)[0]