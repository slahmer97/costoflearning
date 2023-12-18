import random

import torch.random
from omegaconf import DictConfig
import hydra
from collections import deque
from src.meta_network.greedy_policy import GreedyPolicy
from src.meta_network.mcrl import DQN
from src.meta_network.netqueue import ExperienceQueue, ExperiencePriorityQueue
from src.meta_network.policy_selector import PolicySelector, get_policy_selector
from src.meta_network.replay_buffer import ReplayBuffer0
from src.meta_network.tasks import Tasks
from src.simulation import Simulation
import numpy as np
import csv


@hydra.main(config_path="../config", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    task_sampler = Tasks(cfg["traffic"])
    simulator = Simulation()

    learning_queue_config = {
        "sim": simulator,
        "init": 0,
        "queue_type": "fifo"
    }
    forwarded_data = 0
    current_step = 0
    episodes = 500
    steps_per_episode = 1000
    tasks_range = range(1, 2)
    header = ["episode", "reward", "q1", "q2", "rejection1", "drop2", "latency2", "tde", "rho"]
    for task_id in tasks_range:
        filename = f"./data/dynamic-task-{task_id}-0.csv"
        fo = open(filename, "w")
        writer = csv.writer(fo)
        writer.writerow(header)
        task = task_sampler.get_task(task_id)
        task["cost_weights"] = [0.5, 0.5]
        greedy_policy = GreedyPolicy()  # OnlyLearningPlanePolicy() #GreedyPolicy()
        policy_selector = get_policy_selector(**cfg["policy_selector"], task_id=task_id)
        buffer = ReplayBuffer0(input_shape=6, max_size=10000, batch_size=cfg["agent"]["batch_size"])
        learning_queue = ExperienceQueue(**learning_queue_config)
        dqn = DQN(**cfg["agent"])
        #dqn.policy.set_end()
        #policy_selector.set_end()
        simulator.move_environment(**task)
        track_td_err = deque(maxlen=10)
        track_q1_drp = deque(maxlen=1000)
        track_q2_drp = deque(maxlen=1000)
        track_q2_latency = deque(maxlen=1000)
        dqn.eval_net.load_me("./model-history/dynamic-e-task-{}-episode{}".format(0, 70))
        dqn.target_net.load_me("./model-history/dynamic-t-task-{}-episode{}".format(0, 70))
        for episode in range(episodes):
            #if episode == 3 or (task_id == 0 and episode == 0):
            #    greedy_policy = GreedyPolicy()  # OnlyLearningPlanePolicy() #GreedyPolicy()
            #    policy_selector = get_policy_selector(**cfg["policy_selector"])
            #    dqn = DQN(**cfg["agent"])
            if episode % 10 == 0:
                dqn.eval_net.save_me("./model-history/dynamic-e-task-{}-episode{}".format(task_id, episode))
                dqn.target_net.save_me("./model-history/dynamic-t-task-{}-episode{}".format(task_id, episode))
#            dqn.eval_net.save_me("./model-history/dynamic-e-task-{}-episode{}".format(task_id, episode))
#            dqn.target_net.save_me("./model-history/dynamic-t-task-{}-episode{}".format(task_id, episode))

            cum_reward = 0
            loss = 0
            ql_bandwidth = 0
            td_error_sum = 0.0
            td_counter = 0
            for step in range(steps_per_episode):

                ret, rew, additional_learning_res = simulator.rollout(dnn_policy=dqn, greedy_policy=greedy_policy,
                                                                      policy_selector=policy_selector, k_steps=1,
                                                                      meta_data={"cp": learning_queue})
                cum_reward += rew
                for (si, a, (r, c1, c2), sj, _, info) in ret:
                    if a != 3:
                        # drop_sample_prob = float(len(learning_queue)) / 500.0
                        # if np.random.rand() >= drop_sample_prob:
                        td_error = dqn.compute_td_error(si, a, r, sj)
                        td_error_sum += td_error
                        td_counter += 1
                        learning_queue.push((si, a, r, sj), error=td_error)
                        policy_selector.push_stats(td_error=td_error)
                    # push stats
                    track_q1_drp.append(info["packet_drop"][0])
                    track_q2_drp.append(info["packet_dead"][1])
                    track_q2_latency.append(info["packet_latency"][1])
                # additional_learning_res = 100
                samples_tobe_forwarded = learning_queue.step(additional_resources=additional_learning_res)
                ql_bandwidth += len(samples_tobe_forwarded)
                for (si, a, r, sj) in samples_tobe_forwarded:
                    forwarded_data += 1
                    buffer.add(si, a, r, sj)

                if buffer.is_sufficient() and episode < 125:  # and episode < 100:
                    loss += dqn.learn(memory=buffer)
                current_step += 1
            q1 = len(simulator._env.slices[0])
            q2 = len(simulator._env.slices[1])
            rejection1 = np.mean(track_q1_drp)
            drop2 = np.mean(track_q2_drp)
            latency2 = np.mean(track_q2_latency)
            line = [episode, cum_reward, q1, q2, rejection1, drop2, latency2, td_error_sum, policy_selector.rho]
            writer.writerow(line)
            track_td_err.append(td_error_sum)
            print(
                f"Episode={episode}: -- rho={policy_selector.rho}\n"
                f"\tcum_reward={(cum_reward + 250.0) / 250.0} -- epsilon={dqn.policy.epsilon} -- td-error={np.mean(track_td_err)}\n"
                f"\tql_bandwidth={ql_bandwidth} -- ql={len(learning_queue)}\n"
                f"\tq1={q1} -- drp={np.mean(track_q1_drp)}\n"
                f"\tq2={q2} -- dead={np.mean(track_q2_drp)} -- latency={np.mean(track_q2_latency)}\n")
            fo.flush()
        fo.close()


if __name__ == "__main__":
    main()
