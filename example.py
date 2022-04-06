from VNE_env.vne_env import VNEEnv
from VNE_env.param import set_params
from scheduler.base_scheduler import BaseScheduler
from scheduler.ff_scheduler import FirstFitScheduler
from scheduler.bf_scheduler import BestFitScheduler
from scheduler.grc_scheduler import GRCScheduler
from scheduler.gcn_scheduler import GCNScheduler
from scheduler.random_scheduler import RandScheduler
from scheduler.wf_scheduler import WorstFitScheduler
from argparse import Namespace

def run(args: Namespace, scheduler: BaseScheduler):
    env = VNEEnv(args)
    obs = env.reset()
    done = False
    reward_all = 0

    while not done:
        action = scheduler.get_action(obs)
        obs, reward, done, is_reset = env.step(action)
        if is_reset:
            scheduler.reset()
        reward_all += reward

    print(env.log, reward_all)

def main():
    args = set_params()
    args.sn_erdos_renyi = True
    args.vn_erdos_renyi = True

    scheduler_list = [
        GRCScheduler(args),
        FirstFitScheduler(args),
        BestFitScheduler(args),
        WorstFitScheduler(args),
        RandScheduler(args),
        GCNScheduler(args, 3, 64, 100) # 100 is number of substrate graph nodes
    ]
    for scheduler in scheduler_list:
        score = run(args, scheduler)

if __name__ == '__main__':
    main()