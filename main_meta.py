import random
import torch
import copy

from meta_network.greedy_policy import OnlyLearningPlanePolicy, GreedyPolicy, PolicySelector
from meta_network.mcrl import DQN
from meta_network.replay_buffer import ReplayBuffer, ReplayBuffer0
from meta_network.tasks import Tasks
from meta_network.utils import context_to_id, sample_tasks
from simulation import Simulation
import numpy as np
from matplotlib import pyplot as plt
from collections import deque


def single_task():
    task_sampler = Tasks()
    device = torch.device("cpu")
    episodes = 120
    steps_per_episode = 1000
    dqn_config = {
        "exploration_strategy": "epsilon_greedy",
        "temperature": 0.3,
        "mem_capacity": 10000,
        "epsilon": 1.0,
        "epsilon_decay": 0.9995,
        "epsilon_min": 0.01,
        "nn_update": 50,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "learning_rate2": 0.00001,
        "gamma": 0.95,
        "action_count": 3,
        "state_count": 6,
        "continous_ctx_count": 8,
        # "discreet_ctx_count": env_config["max_users:0"] * env_config["max_users:1"],
        "discrete_rep_size": 3,
        # "max_embedding_index": len(mapper)

    }
    device = torch.device("cpu")
    dqn = DQN(**dqn_config)
    task = task_sampler.get_task(0)
    env = Simulation(**task)
    buffer = ReplayBuffer(env_obs_size=3 + 8 + 6, capacity=10000,
                          batch_size=dqn_config["batch_size"], device=device)
    for episode in range(episodes):
        cum_reward = 0
        loss = 0
        for step in range(steps_per_episode):

            ret, rew = env.rollout(dqn, 1)
            cum_reward += rew
            for (si, a, (r, c1, c2), sj, _, _) in ret:
                buffer.add(si, a, r, sj)

            if buffer.idx > dqn_config["batch_size"] * 2:
                loss += dqn.learn(memory=buffer)

        print(f"reward={cum_reward} loss={loss} epsilon={dqn.epsilon}")


def meta_learn():
    task_sampler = Tasks()
    tasks_state = {}
    dqn_config = {
        "exploration_strategy": "epsilon_greedy",
        "temperature": 0.3,
        "mem_capacity": 10000,
        "epsilon": 1.0,
        "epsilon_decay": 0.9995,
        "epsilon_min": 0.01,
        "nn_update": 50,
        "batch_size": 128,
        "learning_rate": 0.0001,
        "learning_rate2": 0.001,
        "gamma": 0.95,
        "action_count": 3,
        "state_count": 6,
        "continous_ctx_count": 8,
        # "discreet_ctx_count": env_config["max_users:0"] * env_config["max_users:1"],
        "discrete_rep_size": 3,
        # "max_embedding_index": len(mapper)

    }
    greedy_policy = GreedyPolicy()
    ps_config = {
        "g_eps": 0.2,
        "g_eps_decay": 0,
        "g_eps_min": 0.002,
        "use": False,
        "end": 80
    }
    policy_selector = PolicySelector(g_eps=ps_config["g_eps"], g_eps_decay=ps_config["g_eps_decay"],
                                     g_eps_min=ps_config["g_eps_min"], use=ps_config["use"])
    policy_selector.reset_gepsilon(ps_config["g_eps"], ps_config["end"])

    iterations = 10000
    rollout_k = 1
    device = torch.device("cpu")
    dqn = DQN(**dqn_config)

    learning_iterations = 1
    glob_rewards = deque(maxlen=50)
    task_rewards = deque(maxlen=50)
    for iteration in range(iterations):
        tasks = [task_sampler.get_task(0)]
        for task in tasks:
            task_id = mapper[(task["max_users:0"], task["max_users:1"], task["resources_count"])]
            if task_id not in tasks_state:
                tasks_state[task_id] = {}
            if "buffer" not in tasks_state[task_id]:
                tasks_state[task_id]["buffer"] = ReplayBuffer0(input_shape=3 + 8 + 6, max_size=10000,
                                                               batch_size=dqn_config["batch_size"])

            if "simulator" not in tasks_state[task_id]:
                tasks_state[task_id]["simulator"] = Simulation(**task)
            tasks_state[task_id]["agent"] = dqn  # copy.deepcopy(dqn)

            tbuff = tasks_state[task_id]["buffer"]
            tsim = tasks_state[task_id]["simulator"]
            tdqn = tasks_state[task_id]["agent"]
            rewall = 0
            for step in range(1000):

                ret, rew, additional_learning_res = tsim.rollout(dnn_policy=tdqn, k_steps=rollout_k,
                                                                 greedy_policy=greedy_policy,
                                                                 policy_selector=policy_selector,
                                                                 meta_data={"cp": []})
                for (si, a, (r, c1, c2), sj, _, _) in ret:
                    tbuff.add(si, a, r, sj)

                #assert rollout_k >= dqn_config["batch_size"]

                value1 = 0
                if tbuff.is_sufficient():
                    value1 += tdqn.learn(memory=tbuff)
                value1 /= learning_iterations

                # ret, rew = tsim.rollout(tdqn, rollout_k)
                # for (si, a, (r, c1, c2), sj, _, _) in ret:
                #    tbuff.add(si, a, r, sj)
                rewall += rew
            glob_rewards.append(rewall)
            value2 = tdqn.meta_learn(tbuff, zero=True)
            task_rewards.append(rewall)

        print(
            f"iter:{iteration} loss1={value1} reward1={np.mean(glob_rewards)} loss2={value2} reward2={np.mean(task_rewards)} buff_size={tbuff.mem_cntr} eps={tdqn.epsilon}")
        # Merge gradients into the global network
        # print("Here")
        dqn.optimizer.zero_grad()
        # for netGlob, netTask in zip(dqn.eval_net.named_parameters(), tdqn.eval_net.named_parameters()):
        #    netGlob[1].grad = netTask[1].grad.clone()
        # print("here")

        # dqn.optimizer2.step()
        dqn.eval_net.load_state_dict(tdqn.eval_net.state_dict())
        dqn.target_net.load_state_dict(dqn.eval_net.state_dict())


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    # sample_tasks()
    # exit(0)
    mapper = context_to_id(200, 200, 15)

    meta_learn()
