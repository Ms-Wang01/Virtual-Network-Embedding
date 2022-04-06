from argparse import Namespace
from typing import List


class RewardCalculator(object):
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.virt_graphs = set()
        self.prev_time = 0
        self.get_reward = self.get_time_reward

    def get_time_reward(self, virt_graphs: List, curr_time: int) -> float:
        reward = 0

        for virt_graph in virt_graphs:
            self.virt_graphs.add(virt_graph)

        for virt_graph in list(self.virt_graphs):
            reward -= (
                min(virt_graph.graph['completion_time'], curr_time) -
                max(virt_graph.graph['arrive_time'], self.prev_time)
            )
            if virt_graph.graph['completed']:
                self.virt_graphs.remove(virt_graph)

        self.prev_time = curr_time

        return reward

    def reset(self) -> None:
        self.virt_graphs.clear()
        self.prev_time = 0
