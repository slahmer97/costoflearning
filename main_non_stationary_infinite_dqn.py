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

    drop_rate = run_config["drop_rate"]
    dqn_config = {
        "exploration_strategy": run_config["exploration_strategy"],
        "temperature": run_config["temperature"],

        "mem_capacity": run_config["mem_capacity"],
        "epsilon": run_config["epsilon"],
        "epsilon_decay": run_config["epsilon_decay"],
        "epsilon_min": run_config["epsilon_min"],
        "nn_update": run_config["nn_update"],
        "batch_size": run_config["batch_size"],
        "learning_rate": run_config["learning_rate"],
        "gamma": run_config["gamma"],
        "action_count": action_count,
        "state_count": state_count,
    }

    dqn = DQN(**dqn_config)

    max_steps = 5000000
    episodes = 5000
    steps_per_episode = 1000

    learning_queue = ExperienceQueue(init=run_config["learning_resources_count"], queue_type=run_config["queue_type"])

    state = env.reset()
    # plotter = StatCollector()

    # plotter.push_data(q1=state[2], q2=state[3], r1=state[4], r2=state[5])
    forwarded_samples = 0.0
    all_generated_samples = 0.0
    step = 0

    wandb.log(
        {
            "slice0:queue-size": state[2],
            "slice0:resources": state[4],
            # "slice0:incoming-traffic": 0,
            "slice0:packet-drop": 0,
            "slice0:packet-dead": 0,

            "slice1:queue-size": state[3],
            "slice1:resources": state[5],
            # "slice1:incoming-traffic": 0,
            "slice1:packet-drop": 0,
            "slice1:packet-dead": 0,

            # "learner:performance": state[2],
            "learner:epsilon": run_config["epsilon"],

        },
        step=step
    )

    reward_list = collections.deque(maxlen=100)

    greedySelector = GreedyBalancer(g_eps=run_config["g_eps"], g_eps_decay=run_config["g_eps_decay"],
                                    g_eps_min=run_config["g_eps_min"])
    dqn.eval_net.load_me("models/eval{}".format(600))
    dqn.target_net.load_me("models/target{}".format(600))
    for i in range(episodes):
        if i in [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]:
            env.move_system()
        accumulated_reward = 0
        queue_learner = []
        dqn.greed_actions = 0
        dqn.non_greedy_action = 0

        greedySelector.greedy = 0
        greedySelector.non_greedy = 0
        q = None
        q_count = 0
        # if i % 100 == 0:
        #    dqn.eval_net.save_me("models/eval{}".format(i))
        #    dqn.target_net.save_me("models/target{}".format(i))

        for j in range(steps_per_episode):

            G.update_success_prob(dqn, learning_queue, greedySelector)

            step += 1

            ql_size = len(learning_queue)
            queue_learner.append(ql_size)

            action, action_type, q_vals = dqn.choose_action(state)

            if q_vals is not None:
                if q is not None:
                    q += q_vals
                    q_count += 1
                else:
                    q = q_vals
                    q_count = 1

            if i + j > 0 and run_config["use_greedy"]:
                apply, greedy_selection = greedySelector.act_int_pg(CP=len(learning_queue),
                                                                    UP=(
                                                                        int(info["packet_urgent"][0]),
                                                                        int(info["packet_urgent"][1]) + 2),
                                                                    DP=(int(info["queue_size"][0]),
                                                                        int(info["queue_size"][0])),
                                                                    E=(info['incoming_traffic'][0],
                                                                       info['incoming_traffic'][1]))
            else:
                apply = False

            if apply:
                next_state, reward, done, info = env.step(3, greedy_selection)
                # print("{}".format(reward))
            else:
                next_state, reward, done, info = env.step(action)

            all_generated_samples += 1

            def probDropLearningSample(lqueue, action_type, eps):
                return 0

            new_drp_rate = len(learning_queue) / 1500.0

            # if drop_rate <= np.random.random(1)[0] and not apply:
            if run_config["use_prob_selection"] and new_drp_rate <= np.random.random(1)[0] and not apply:
                learning_queue.push((state, action, reward, next_state))
            elif not run_config["use_prob_selection"]:
                learning_queue.push((state, action, reward, next_state))

            learner_throughput = 0.0
            if apply:
                samples = learning_queue.step(additional_resources=greedy_selection[2])
                learner_throughput = len(samples)
                forwarded_samples += len(samples)
                for (si, a, r, sj) in samples:
                    dqn.store_transition(si, a, r, sj)

                    # changed from run_config to a fixed thing
                    if dqn.memory_counter >= run_config["batch_size"] * 15:
                        dqn.learn()
            else:
                if run_config["use_greedy"]:
                    learning_queue.last_resource_usage = 0
                else:
                    samples = learning_queue.step()
                    learner_throughput = len(samples)
                    forwarded_samples += len(samples)
                    for (si, a, r, sj) in samples:
                        dqn.store_transition(si, a, r, sj)

                        # changed from run_config to a fixed thing
                        if dqn.memory_counter >= run_config["batch_size"] * 15:
                            dqn.learn()
            assert info["resources"][0] + info["resources"][1] <= 15

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
                lrho=greedySelector.g_eps,
                lepsilon=dqn.epsilon
            )

            state = next_state
            wandb.log(
                {
                    "learner:resources": learning_queue.last_resource_usage
                },
                step=step
            )

            if step % 10 == 0:
                wandb.log(
                    {
                        "slice0:queue-size": info["queue_size"][0],
                        "slice0:resources": info["resources"][0],
                        "slice0:incoming-traffic": info['incoming_traffic'][0],
                        "slice0:packet-drop": info['packet_drop'][0],
                        "slice0:packet-dead": info['packet_dead'][0],
                        "slice0:urgent-packet-count": info["packet_urgent"][0],
                        "slice0:resents": info["packet_resent"][0],
                        "slice0:latency_per_packet": info["packet_latency"][0],

                        "slice1:queue-size": info["queue_size"][1],
                        "slice1:resources": info["resources"][1],
                        "slice1:incoming-traffic": info['incoming_traffic'][1],
                        "slice1:packet-drop": info['packet_drop'][1],
                        "slice1:packet-dead": info['packet_dead'][1],
                        "slice1:urgent-packet-count": info["packet_urgent"][1],
                        "slice1:resents": info["packet_resent"][1],
                        "slice1:latency_per_packet": info["packet_latency"][1],

                        "learner:queue-size": ql_size,
                        "learner:forwarded-experiences": forwarded_samples,
                        "learner:greedySelectionEPS": greedySelector.g_eps,

                    },
                    step=step
                )

            accumulated_reward += reward

        print("[+] Step:{}k -- Greedy: {} - ({}) -- Non-Greedy: {} - ({})".format(i, greedySelector.greedy,
                                                                                  env.reward_greedy,
                                                                                  greedySelector.non_greedy,
                                                                                  env.reward_non_greedy))
        reward_list.append(accumulated_reward)
        env.reward_non_greedy = 0
        env.reward_greedy = 0
        if q is not None:
            # q /= q_count

            wandb.log(
                {
                    "learner:performance": np.mean(reward_list),
                    "learner:epsilon": dqn.epsilon,
                    "q:0": q[0],
                    "q:1": q[1],
                    "q:2": q[2]
                },
                step=step
            )
        else:
            wandb.log(
                {
                    "learner:performance": np.mean(reward_list),
                    "learner:epsilon": dqn.epsilon,

                },
                step=step
            )

        if step >= max_steps:
            return


# 182.6
def run_experiments():
    c = {
        "use_prob_selection": [True],
        "use_greedy": [True],
        "queue_type": ["lifo"],
        # "max_users:0": [15, 28, 25, 7, 20, 15, 19, 18, 15, 36],
        # "max_users:1": [19, 10, 12, 25, 16, 20, 17, 17, 19, 5],
        "max_users:0": [21, 28, 24, 14],
        "max_users:1": [18, 13, 16, 23],
        "network_resources_count": [15],
        "learning_resources_count": [0]
    }

    strategy = "epsilon_greedy"
    for i in range(0, len(c["use_greedy"])):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        config = {
            "sim-id": "Dynamic500kLoaded",
            "use_prob_selection": c["use_prob_selection"][i],
            "use_greedy": c["use_greedy"][i],
            "queue_type": c["queue_type"][i],
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

            "network_resources_count": c["network_resources_count"][i],
            "learning_resources_count": c["learning_resources_count"][i],

            "max_users:0": c["max_users:0"],
            "max_users:1": c["max_users:1"],

            "drop_rate": 0.0,

            "p_success": 1.0,

        }

        wandb.init(reinit=True, config=config, project="costoflearning-test")
        wandb.run.name = "Sim={}/{},u={}/{},useG={}".format(c["network_resources_count"][i],
                                                            c["learning_resources_count"][i],
                                                            c["max_users:0"][i],
                                                            c["max_users:1"][i], 0.0,
                                                            config["use_greedy"])
        wandb.run.save()

        run(**config)

        wandb.finish()


run_experiments()
