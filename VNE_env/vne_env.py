import sys
import torch
import numpy as np
import networkx as nx
from argparse import Namespace
from VNE_env.timeline import Timeline
from torch_geometric.data import Data
from VNE_env.wall_time import WallTime
from VNE_env.check_valid import check_done
from VNE_env.check_valid import check_action
from VNE_env.check_valid import check_observe
from VNE_env.check_valid import check_schedule
from VNE_env.reservation_info import Reservation
from VNE_env.graph import generate_substrate_graph
from VNE_env.reward_calculator import RewardCalculator
from VNE_env.schedule_policy import shortest_job_first
from VNE_env.job_generator import generate_virtual_graphs

class VNEEnv(object):
    def __init__(self, args: Namespace):
        self.args = args
        self.debug_info = []
        
        if sys.version_info < (3, 8):
            self.make_error('Python version must be >= 3.8')

        self.seeding(args.seed)
        self.wall_time = WallTime()
        self.timeline = Timeline()
        self.sched_policy = shortest_job_first
        self.reset_time = 0

        self.reward_calculator = RewardCalculator(args)

    def seeding(self, seed):
        self.seed = seed
        rng = np.random.RandomState()
        rng.seed(seed)
        self.rng = rng
    
    def setup_substrate_graph(self):
        self.subs_graph = generate_substrate_graph(self.args, self.rng)

    def setup_virtual_graphs(self):
        self.virt_graphs, not_arrived_vgs = \
            generate_virtual_graphs(self.args, self.rng) 

        for t, virtual_graph in not_arrived_vgs:
            self.timeline.push(t, virtual_graph)

        self.scheduled_set = set()
        self.finished_virtual_graphs = set()

    def setup_space(self):
        args = self.args
    
        self.vn_low = np.array(
            [args.vn_feat_low] * 
            args.node_num_res +
            [0] * 2
        )

        self.vl_low = np.array(
            [args.vl_feat_low] * 
            args.link_num_res
        )

        self.vn_high = np.array(
            [args.vn_feat_high] * 
            args.node_num_res +
            [np.inf, 1]
        )

        self.vl_high = np.array(
            [args.vl_feat_high] *
            args.link_num_res
        )

        self.sn_low = np.zeros(
            (args.node_num_res * 3 + 2,)
        )

        self.sl_low = np.zeros((args.link_num_res,))
        
        sn_feats = np.zeros((
            len(self.subs_graph.nodes), 
            self.args.node_num_res
        ))

        sl_feats = np.zeros((
            len(self.subs_graph.edges),
            self.args.link_num_res
        ))

        for subs_node in self.subs_graph.nodes:
            sn_feats[subs_node] += \
                self.subs_graph.nodes[subs_node]['features']
    
        for edge in self.subs_graph.edges:
            sl_feat = self.subs_graph.edges[edge]
            sl_feats[sl_feat['idx']] += sl_feat['features']

        sn_left = np.concatenate([sn_feats] * 3, axis=1)
        sn_right = np.array([[np.inf] * 2] * sn_left.shape[0])

        self.sn_high = np.concatenate((sn_left, sn_right), axis=1)
        self.sl_high = sl_feats


    def observe(self):
        # pytorch version
        virt_graph = self.reservation.virt_graph

        cur_subs_node_feats = np.zeros((
            len(self.subs_graph.nodes), 
            self.args.node_num_res
        ))
        subs_node_types = np.zeros((len(self.subs_graph.nodes),))

        subs_links = np.zeros((len(self.subs_graph.edges), 2))
        subs_link_feats = np.zeros((len(self.subs_graph.edges), self.args.link_num_res))
        subs_link_types = np.zeros((len(self.subs_graph.edges),))

        for subs_node in self.subs_graph.nodes:
            subs_node_val = self.subs_graph.nodes[subs_node]
            cur_subs_feat = subs_node_val['features']
            cur_subs_node_feats[subs_node] += cur_subs_feat
            subs_node_types[subs_node] = subs_node_val['type']
            
        for subs_link in self.subs_graph.edges:
            subs_link_val = self.subs_graph.edges[subs_link]
            idx = subs_link_val['idx']
            subs_links[idx] = subs_link
            subs_link_feats[idx] = subs_link_val['features']
            subs_link_types[idx] = subs_link_val['type']

        virt_node_feats = np.zeros((
            len(virt_graph.nodes), 
            self.args.node_num_res
        ))

        virt_schedulable_mask = np.array(
            [True] * len(virt_graph.nodes)
        )

        virt_links = np.zeros((len(virt_graph.edges), 2))
        virt_link_feats = np.zeros((len(virt_graph.edges), self.args.link_num_res))
        virt_runtimes = np.array([virt_graph.graph['runtime']] * len(virt_graph.nodes))


        for virt_node in virt_graph.nodes:
            virt_node_val = virt_graph.nodes[virt_node]
            virt_feat = virt_node_val['features']
            virt_node_feats[virt_node] += virt_feat
            virt_schedulable_mask[virt_node] = \
                virt_node not in self.reservation.virt_tracker
                
        for virt_link in virt_graph.edges:
            virt_link_val = virt_graph.edges[virt_link]
            idx = virt_link_val['idx']
            virt_links[idx] = virt_link
            virt_link_feats[idx] = virt_link_val['features']
        
        sch_subs_node_feats = np.zeros((
            len(self.subs_graph.nodes), 
            self.args.node_num_res
        ))
        subs_node_sch_virt = np.zeros((len(self.subs_graph.nodes),))

        for schedule in self.scheduled_set:
            for subs_node, node_feat in \
                schedule.subs_node_feat_tracker.items():
                sch_subs_node_feats[subs_node] += node_feat

            for subs_node, virt_nodes in \
                schedule.subs_node_tracker.items():
                subs_node_sch_virt[subs_node] = len(virt_nodes)
            
        res_subs_node_feats = np.zeros((
            len(self.subs_graph.nodes), 
            self.args.node_num_res
        ))

        subs_node_res_virt = np.zeros((len(self.subs_graph.nodes),))
        
        for subs_node, virt_nodes in \
            self.reservation.subs_node_tracker.items():
            subs_node_res_virt[subs_node] += len(virt_nodes)

        for subs_node, node_feat in \
            self.reservation.subs_node_feat_tracker.items():
            res_subs_node_feats[subs_node] += node_feat

        subs_graph_node_x = np.concatenate((
            cur_subs_node_feats,
            sch_subs_node_feats,
            res_subs_node_feats,
            subs_node_sch_virt.reshape(-1, 1),
            subs_node_res_virt.reshape(-1, 1),
        ), axis=1)

        subs_graph_edge_index = subs_links.T
        subs_graph_edge_type = subs_link_types
        subs_graph_edge_attr = subs_link_feats

        virt_graph_sched_x = virt_schedulable_mask.reshape(-1, 1)
        
        virt_graph_node_x = np.concatenate((
            virt_node_feats,
            virt_runtimes.reshape(-1, 1),
            virt_graph_sched_x
        ), axis=1)

        virt_graph_edge_index = virt_links.T
        virt_graph_edge_attr = virt_link_feats

        subs_graph_data = Data()
        subs_graph_data.x = torch.from_numpy(subs_graph_node_x).to(torch.float32)
        subs_graph_data.edge_attr = torch.from_numpy(subs_graph_edge_attr).to(torch.float32)
        subs_graph_data.edge_type = torch.from_numpy(subs_graph_edge_type).to(torch.float32)
        subs_graph_data.edge_index = torch.from_numpy(subs_graph_edge_index).to(torch.int64)

        if self.args.no_same_place:
            subs_scheduable_mask = subs_node_res_virt == 0
            subs_graph_data.mask = torch.from_numpy(subs_scheduable_mask).to(torch.bool)

        virt_graph_data = Data()
        virt_graph_data.x = torch.from_numpy(virt_graph_node_x).to(torch.float32)
        virt_graph_data.mask = torch.from_numpy(virt_schedulable_mask).to(torch.bool)
        virt_graph_data.edge_attr = torch.from_numpy(virt_graph_edge_attr).to(torch.float32)
        virt_graph_data.edge_index = torch.from_numpy(virt_graph_edge_index).to(torch.int64)
        virt_graph_data.idx = virt_graph.graph['idx']

        observe = virt_graph_data, subs_graph_data

        if not check_observe(self, observe):
            self.make_error('Unexpected observation')

        return observe

    def make_reservation(self, action):
        virt_node, subs_node = action
        return self.reservation.add(virt_node, subs_node)
        
    def reset_reservation(self): 
        reservation = self.reservation
        self.reservation = None
        self.recover_substrate_graph(reservation)

    def recover_substrate_graph(self, reservation):        
        virt_graph = reservation.virt_graph
        virt_track = reservation.virt_tracker
        subs_link_track = reservation.subs_link_tracker

        for virt_node, subs_node in virt_track.items():
            self.subs_graph.nodes[subs_node]['features'] += \
                virt_graph.nodes[virt_node]['features']

        for (src, dst), flow in subs_link_track.items():
            if src < dst: # avoid double-recover
                self.subs_graph.edges[(src, dst)]['features'] += flow

    def select_virtual_graph_to_schedule(self) -> bool:
        virt_graph = self.sched_policy(self)

        if virt_graph is None:
            return

        self.debug_info.append(f'vne_env@svgts: {virt_graph.graph["idx"]} is selected')

        self.log['try_scheduled'] += 1
        
        reservation = Reservation(self.args, virt_graph, self.subs_graph)
        self.reservation = reservation

    def schedule(self) -> None:
        schedule = self.reservation
    
        if not check_schedule(self, schedule):
            self.make_error('Unreliable schedule information')

        virt_graph = schedule.virt_graph
        self.reservation = None

        done_time = self.wall_time.curr_time + virt_graph.graph['runtime']

        virt_graph.graph['scheduled'] = True
        virt_graph.graph['scheduled_time'] = self.wall_time.curr_time

        self.scheduled_set.add(schedule)
        self.timeline.push(done_time, schedule)

    def process_timeline(self):
        new_time, obj = self.timeline.pop()
        self.wall_time.update_time(new_time)

        if isinstance(obj, Reservation):
            # scheduled virtual graph is done
            reservation = obj

            virt_graph = reservation.virt_graph
            virt_graph.graph['completed'] = True
            virt_graph.graph['completion_time'] = new_time

            self.scheduled_set.remove(reservation)
            self.recover_substrate_graph(reservation)
            self.virt_graphs.remove(virt_graph)
            self.finished_virtual_graphs.add(virt_graph)
        elif isinstance(obj, nx.DiGraph):
            # new virtual graph is comming
            virt_graph = obj
            if virt_graph.graph['completed']:
                self.make_error('Virtual graph already completed')
            if virt_graph.graph['scheduled']:
                self.make_error('Virtual graph already scheduled')
            if virt_graph.graph['arrived']:
                self.make_error('Virtual graph already arrived')

            virt_graph.graph['arrived'] = True
            
            self.virt_graphs.add(virt_graph)
        else:
            raise NotImplementedError

        return obj

    def step(self, action):
        virt_node, subs_node = action

        self.debug_info.append(f'vne_env@step: assign {virt_node} to {subs_node}')
        
        if not check_action(self, virt_node, subs_node):
            self.make_error('Invalid action')

        is_reset = True
        if not self.make_reservation(action):
            # newtork failure process -> 
            # reset reservation
            self.log['network_failed'] += 1
            self.reset_reservation()
        elif self.reservation.schedulable():
            # schedule process -> 
            # scheduling -> find new schedule
            self.schedule()
            self.select_virtual_graph_to_schedule()
        elif not self.reservation.can_reserve():
            # node failure process -> 
            # reset reservation
            self.log['node_failed'] += 1
            self.reset_reservation()
        else:
            # continue scheduling -> nothing changed.
            is_reset = False
        
        # timelining
        while 0 < len(self.timeline):
            # if reservation existed, no timelining
            if self.reservation is not None:
                break
            self.process_timeline()
            self.select_virtual_graph_to_schedule()
        
        reward = self.reward_calculator.get_reward(
            self.virt_graphs, 
            self.wall_time.curr_time
        )

        if (
            self.reservation is None and 
            0 < len(self.timeline)   and
            0 < len(self.virt_graphs)
        ):
            self.make_error('Invalid step situation')

        done = self.reservation is None

        if done:
            if not check_done(self):
                self.make_error('Final state is invalid.')
            return None, reward, done, is_reset
        else:
            return self.observe(), reward, done, is_reset

    def make_error(self, error_msg):
        print('args:', self.args)
        print('reset_time:', self.reset_time)
        print('error_logs: ------------------------------------')
        print(*self.debug_info, sep='\n')
        print('------------------------------------------------')
        raise Exception(error_msg)

    def reset(self) -> None:
        self.reset_time += 1
        self.debug_info = []

        self.log = {
            'node_failed': 0,
            'network_failed': 0,
            'try_scheduled': 0
        }

        self.wall_time.reset()
        self.timeline.reset()
        self.reward_calculator.reset()

        self.setup_substrate_graph()
        self.setup_virtual_graphs()
        self.setup_space()

        self.select_virtual_graph_to_schedule()


        return self.observe()