import random
import torch
import copy
from meta_network.mcrl import DQN
from meta_network.replay_buffer import ReplayBuffer
from meta_network.tasks import Tasks
from meta_network.utils import context_to_id, sample_tasks
from simulation import Simulation
import numpy as np
from matplotlib import pyplot as plt

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# sample_tasks()
# exit(0)

mapper = context_to_id(200, 200, 15)
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
    "batch_size": 64,
    "learning_rate": 0.0001,
    "gamma": 0.95,
    "action_count": 3,
    "state_count": 6,
    "continous_ctx_count": 8,
    # "discreet_ctx_count": env_config["max_users:0"] * env_config["max_users:1"],
    "discrete_rep_size": 3,
    # "max_embedding_index": len(mapper)

}

iterations = 100
rollout_k = 1000
device = torch.device("cpu")
dqn = DQN(**dqn_config)

learning_iterations = 10

for iteration in range(iterations):
    tasks = task_sampler.sample_batch_tasks(1)
    for task in tasks:
        task_id = mapper[(task["max_users:0"], task["max_users:1"], task["resources_count"])]
        if task_id not in tasks_state:
            tasks_state[task_id] = {}
        if "buffer" not in tasks_state[task_id]:
            tasks_state[task_id]["buffer"] = ReplayBuffer(env_obs_size=3 + 8 + 6, capacity=100000,
                                                          batch_size=dqn_config["batch_size"], device=device)

        if "simulator" not in tasks_state[task_id]:
            tasks_state[task_id]["simulator"] = Simulation(**task)
        tasks_state[task_id]["agent"] = copy.deepcopy(dqn)

        tbuff = tasks_state[task_id]["buffer"]
        tsim = tasks_state[task_id]["simulator"]
        tdqn = tasks_state[task_id]["agent"]
        ret, rew = tsim.rollout(dqn, rollout_k)
        for (si, a, (r, c1, c2), sj, _, _) in ret:
            tbuff.add(si, a, r, sj)

        assert tbuff.idx > dqn_config["batch_size"] * 2
        for learning_iter in range(learning_iterations):
            tdqn.learn(memory=tbuff)

    # At this point, we should have all Ph_{i} read
