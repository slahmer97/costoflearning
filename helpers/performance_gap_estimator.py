import collections
import random
import numpy as np
import torch

from greedy_selector import GreedyBalancer
from network.netqueue import ExperienceQueue
from network.networkNonStationary import Network
from network.neuralN import DQN
from sim_perf_collector import PerfCollector
from network.globsim import SimGlobals as G

import wandb


def run(**run_config):
    print("[+] Starting a new run with the following configurations:")
    print("{}".format(run_config))
    print()
    G.reset()
    G.RESOURCES_COUNT = run_config["network_resources_count"]
    G.INIT_LEARNING_RESOURCES = run_config["learning_resources_count"]

    G.init_success_prob(run_config["p_success"])

    simStatCollector = PerfCollector(filename=run_config["sim-id"])
    # init the globals
    env_config = {
        "max_users:0": run_config["max_users:0"],
        "max_users:1": run_config["max_users:1"]
    }
    env = Network(**env_config)
    action_count = env.action_space.n
    state_count = env.observation_space.shape[0]
    print(action_count, state_count)
    transfer = run_config["train"]
    dqn = run_config["policy"]

    episodes = 500
    steps_per_episode = 1000

    learning_queue = ExperienceQueue(init=run_config["learning_resources_count"], queue_type=run_config["queue_type"])

    state = env.reset()
    # plotter = StatCollector()

    # plotter.push_data(q1=state[2], q2=state[3], r1=state[4], r2=state[5])
    forwarded_samples = 0.0
    all_generated_samples = 0.0
    step = 0

    reward_list = collections.deque(maxlen=100)

    for i in range(episodes):
        accumulated_reward = 0
        queue_learner = []
        dqn.greed_actions = 0
        dqn.non_greedy_action = 0

        for j in range(steps_per_episode):

            step += 1

            ql_size = len(learning_queue)
            queue_learner.append(ql_size)

            action, action_type, q_vals = dqn.choose_action(state)

            next_state, reward, done, info = env.step(action)

            all_generated_samples += 1

            learner_throughput = 0.0

            if transfer:
                learning_queue.push((state, action, reward, next_state))

                samples = learning_queue.step()
                learner_throughput = len(samples)
                forwarded_samples += len(samples)
                for (si, a, r, sj) in samples:
                    dqn.store_transition(si, a, r, sj)

                    # changed from run_config to a fixed thing
                    if dqn.memory_counter >= run_config["batch_size"] * 15:
                        dqn.learn()

            simStatCollector.push_stats(
                s0throughputs=info["packet_served"][0],
                s1throughputs=info["packet_served"][1],

                s0death_rates=info["packet_dead"][0],
                s1death_rates=info["packet_dead"][1],

                s0drop_rates=info["packet_drop"][0],
                s1drop_rates=info["packet_drop"][1],

                s0queue_sizes=info["queue_size"][0],
                s1queue_sizes=info["queue_size"][1],

                s0urgent_packets=info["packet_urgent"][0],
                s1urgent_packets=info["packet_urgent"][1],

                s0latency_per_packet=info["packet_latency"][0],
                s1latency_per_packet=info["packet_latency"][1],

                s0resent=info["packet_resent"][0],
                s1resent=info["packet_resent"][1],

                link_capacity=G.success_prob,
                lqueue_sizes=ql_size,
                lthroughputs=learner_throughput,
                lrho=0,
                lepsilon=dqn.epsilon
            )

            state = next_state

            accumulated_reward += reward

        reward_list.append(accumulated_reward)
        print("[+] Step:{}k : E[G]={} -- EPS={}".format(i, np.mean(reward_list), dqn.epsilon))

        env.reward_non_greedy = 0
        if not transfer:
            if i == 100:
                return


# 182.6
def run_experiments(id, max_user0, max_user1, train, policy):
    c = {
        "use_prob_selection": False,
        "use_greedy": False,
        "queue_type": "fifo",
        "network_resources_count": 15,
        "learning_resources_count": 10
    }

    strategy = "epsilon_greedy"

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    config = {
        "sim-id": id,
        "use_prob_selection": c["use_prob_selection"],
        "use_greedy": c["use_greedy"],
        "queue_type": c["queue_type"],
        "g_eps": 3.0 / 15.0,
        "g_eps_decay": 0.99998849492,
        "g_eps_min": 0.02,

        "mem_capacity": 10000,
        "gamma": 0.95,
        "epsilon": 1.0,
        "epsilon_decay": 0.9995,
        "epsilon_min": 0.01,
        "nn_update": 50,
        "batch_size": 32,
        "learning_rate": 0.00001,
        "temperature": 1.2,
        "exploration_strategy": strategy,

        "network_resources_count": c["network_resources_count"],
        "learning_resources_count": c["learning_resources_count"],

        "max_users:0": [max_user0],
        "max_users:1": [max_user1],

        "drop_rate": 0.0,

        "p_success": 1.0,
        "policy": policy,
        "train": train,

    }

    run(**config)


def main():
    dqn_config = {
        "exploration_strategy": "epsilon_greedy",
        "temperature": 1.2,
        "mem_capacity": 10000,
        "epsilon": 1.0,
        "epsilon_decay": 0.9995,
        "epsilon_min": 0.01,
        "nn_update": 50,
        "batch_size": 32,
        "learning_rate": 0.00001,
        "gamma": 0.95,
        "action_count": 3,
        "state_count": 6,
    }
    done = False
    for name in ["30-10-300"]:
        candidates = [(27, 13), (26, 13), (25, 13), (25, 14), (24, 13), (24, 14), (24, 15), (23, 13), (23, 14), (23, 15), (22, 13), (22, 14), (22, 15), (22, 16), (21, 14), (21, 15), (21, 16), (21, 17), (20, 15), (20, 16), (20, 17), (19, 15), (19, 16), (19, 17), (19, 18), (18, 16), (18, 17), (18, 18), (18, 19), (17, 17), (17, 18), (17, 19), (17, 20), (16, 17), (16, 18), (16, 19), (16, 20), (15, 18), (15, 19), (15, 20), (15, 21), (14, 19), (14, 20), (14, 21), (14, 22), (13, 19), (13, 20), (13, 21), (13, 22), (12, 20), (12, 21), (12, 22), (12, 23), (11, 21), (11, 22), (11, 23), (11, 24), (10, 22), (10, 23), (10, 24), (9, 22), (9, 23), (9, 24), (9, 25), (8, 23), (8, 24), (8, 25), (8, 26), (7, 24), (7, 25), (7, 26), (7, 27), (6, 24), (6, 25), (6, 26), (6, 27)]


        for u0, u1 in candidates:
            term1 = u0 * 0.35
            term2 = u1 * 0.5

            load = term1 + term2
            if 13 >= load:
                continue
            if load >= 17:
                continue
            dqn_t = DQN(**dqn_config)

            name_experiment = "train-on={}-{}   tested-on={}-{}".format(30, 10, u0, u1)
            run_experiments(name_experiment, u0, u1, True, dqn_t)


if __name__ == '__main__':
    main()