import random
import torch

from meta_network.greedy_policy import GreedyPolicy, PolicySelector, OnlyLearningPlanePolicy
from meta_network.mcrl import DQN
from meta_network.netqueue import ExperienceQueue
from meta_network.replay_buffer import ReplayBuffer, ReplayBuffer0
from meta_network.tasks import Tasks
from meta_network.utils import context_to_id  # sample_tasks
from sim_perf_collector import PerfCollector
from simulation import Simulation
import numpy as np
import wandb


# keep the average queue size for each queue, and use it as a heuristic to allocate the remaining blocks
def single_task():
    task_sampler = Tasks()
    device = torch.device("cpu")
    episodes = 200
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
        "learning_rate": 0.00001,
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
    idd = 110
    task = task_sampler.get_task(idd)
    task["resources_count"] = 15
    # task["max_users:1"] = 21
    # task["max_users:0"] = 12

    coeif = 0.5
    task["cost_weights"] = [coeif, 1.0 - coeif]
    task_id = "{}-{}".format(idd, "dynamic")
    # simStatCollector = PerfCollector(filename=task_id)
    #for i, t in enumerate(task_sampler.tasks):
    #    if t["load"] > 14.75:
    #        print(i, t)
    #print(task_sampler.tasks)
    print(task)
    #exit(0)
    env = Simulation(**task)

    greedy_policy = GreedyPolicy()  # OnlyLearningPlanePolicy() #GreedyPolicy()

    ps_config = {
        "g_eps": 0.2,
        "g_eps_decay": 0,
        "g_eps_min": 0.002,
        "use": True,
        "end": 80
    }

    policy_selector = PolicySelector(g_eps=ps_config["g_eps"], g_eps_decay=ps_config["g_eps_decay"],
                                     g_eps_min=ps_config["g_eps_min"], use=ps_config["use"])
    policy_selector.reset_gepsilon(ps_config["g_eps"], ps_config["end"])

    buffer = ReplayBuffer0(input_shape=3 + 8 + 6, max_size=10000, batch_size=dqn_config["batch_size"])
    # buffer = ReplayBuffer(env_obs_size=3 + 8 + 6, capacity=10000,
    #                      batch_size=dqn_config["batch_size"], device=device)

    learning_queue_config = {
        "sim": env,
        "init": 0,
        "queue_type": "fifo"
    }
    learning_queue = ExperienceQueue(**learning_queue_config)
    forwarded_data = 0
    current_step = 0

    simulation_config = {
        "dqn_config": dqn_config.copy(),
        "task_config": task.copy(),
        "greedy_policy": policy_selector.__class__.__name__,
        "ps_config": ps_config,
        "learning_queue_config": learning_queue_config

    }
    wandb.init(reinit=False, config=simulation_config, project="non-stationary-CoL")
    wandb.run.name = task_id
    wandb.run.save()
    dqn.eval_net.load_me("./ex-model/task{}-episode{}".format(idd, 60))
    dqn.target_net.load_me("./ex-model/task{}-episode{}".format(idd, 60))
    for episode in range(episodes):
        cum_reward = 0
        loss = 0
        avg_queue0 = []
        avg_queue1 = []
        avg_latency1 = []
        avg_active0 = []
        avg_active1 = []
        avg_resource0 = []
        avg_resource1 = []
        #if episode % 10 == 0:
        #    dqn.eval_net.save_me("./ex-model/task{}-episode{}".format(idd, episode))
        #    dqn.target_net.save_me("./ex-model/task{}-episode{}".format(idd, episode))

        for step in range(steps_per_episode):

            ret, rew, additional_learning_res = env.rollout(dnn_policy=dqn, greedy_policy=greedy_policy,
                                                            policy_selector=policy_selector, k_steps=1,
                                                            meta_data={"cp": learning_queue})
            cum_reward += rew
            for (si, a, (r, c1, c2), sj, _, info) in ret:
                if a != 3:
                    # drop_sample_prob = float(len(learning_queue)) / 500.0
                    # if np.random.rand() >= drop_sample_prob:
                    learning_queue.push((si, a, r, sj))

                # avg_queue0.append(info["queue_size"][0])
                # avg_queue1.append(info["queue_size"][1])
                # avg_latency1.append(info["packet_latency"][1])
                # avg_active0.append(info["active_users"][0])
                # avg_active1.append(info["active_users"][1])
                # avg_resource0.append(info["resources"][0])
                # avg_resource1.append(info["resources"][1])
                wandb.log(
                    {
                        # "reward": (cum_reward + 250.0) / 250.0,
                        # "epsilon": dqn.epsilon,
                        # "forwarded_data": forwarded_data,
                        "loss": loss,

                        "drop-0": info["packet_drop"][0],
                        "dead-1": info["packet_dead"][1],

                        "latency-0": info["packet_latency"][0],
                        "latency-1": info["packet_latency"][1],
                        "active-0": info["active_users"][0],
                        "active-1": info["active_users"][1],
                        "queue-0": info["queue_size"][0],
                        "queue-1": info["queue_size"][1],
                        "resources-0": info["resources"][0],
                        "resources-1": info["resources"][1],

                    },
                    step=current_step
                )

            samples_tobe_forwarded = learning_queue.step(additional_resources=additional_learning_res)
            for (si, a, r, sj) in samples_tobe_forwarded:
                # if a != 3:
                forwarded_data += 1
                buffer.add(si, a, r, sj)

            if buffer.is_sufficient() and episode < 101:  # and episode < 100:
                loss += dqn.learn(memory=buffer)
            current_step += 1

        wandb.log(
            {
                "reward": (cum_reward + 300.0) / 300.0,
                "epsilon": dqn.epsilon,
                "forwarded_data": forwarded_data,
                "loss": loss,
                # "latency-1": np.mean(avg_latency1),
                # "active-0": np.mean(avg_active0),
                # "active-1": np.mean(avg_active1),
                # "queue-0": np.mean(avg_queue0),
                # "queue-1": np.mean(avg_queue1),
                # "resources-0": np.mean(avg_resource0),
                # "resources-1": np.mean(avg_resource1),

            },
            step=current_step
        )

        greedy_count, non_greedy_count, greedy_acc, non_greedy_acc = env.get_greedy_info()
        env.reset_greedy_counters()
        greedy_acc = float(greedy_acc)
        non_greedy_acc = float(non_greedy_acc)
        print(
            f"step={episode + 1}k reward={(cum_reward + 250.0) / 250.0} loss={loss} epsilon={dqn.epsilon} forwarded={forwarded_data}/{buffer.mem_cntr} g-eps={policy_selector.g_eps} greedy={greedy_count}/{greedy_acc} non-reedy={non_greedy_count}/{non_greedy_acc} len(learning)={len(learning_queue)}")

    wandb.finish()

def traffic_profile():
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
        "learning_rate": 0.00001,
        "learning_rate2": 0.000001,
        "gamma": 0.95,
        "action_count": 3,
        "state_count": 6,
        "continous_ctx_count": 8,
        # "discreet_ctx_count": env_config["max_users:0"] * env_config["max_users:1"],
        "discrete_rep_size": 3,
        # "max_embedding_index": len(mapper)

    }

    iterations = 10000
    rollout_k = 1000
    dqn = DQN(**dqn_config)
    greedy_policy = OnlyLearningPlanePolicy()
    ps_config = {
        "g_eps": 0.2,
        "g_eps_decay": 0,
        "g_eps_min": 0.002,
        "use": True,
        "end": 80
    }

    policy_selector = PolicySelector(g_eps=ps_config["g_eps"], g_eps_decay=ps_config["g_eps_decay"],
                                     g_eps_min=ps_config["g_eps_min"], use=ps_config["use"])
    policy_selector.reset_gepsilon(ps_config["g_eps"], ps_config["end"])

    current_step = 0
    wandb.init(reinit=False, project="non-stationary-traffic-shape")
    wandb.run.save()
    for i in range(len(task_sampler.tasks)):
        current_task = task_sampler.get_task(i)
        env = Simulation(**current_task)
        learning_queue_config = {
            "sim": env,
            "init": 0,
            "queue_type": "fifo"
        }
        learning_queue = ExperienceQueue(**learning_queue_config)
        for _ in range(100000):
            ret, rew, additional_learning_res = env.rollout(dnn_policy=dqn, greedy_policy=greedy_policy,
                                                            policy_selector=policy_selector, k_steps=1,
                                                            meta_data={"cp": learning_queue})
            for (si, a, (r, c1, c2), sj, _, info) in ret:
                wandb.log(
                    {
                        "active-0": info["active_users"][0],
                        "active-1": info["active_users"][1],
                    },
                    step=current_step
                )
            current_step += 1

    wandb.finish()


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # sample_tasks()
    # exit(0)
    mapper = context_to_id(200, 200, 15)

    traffic_profile()
