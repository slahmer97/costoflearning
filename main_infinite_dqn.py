import collections
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from greedy_selector import GreedyBalancer
from network.netqueue import ExperienceQueue
from network.network import Network
import wandb
from stat_collector import StatCollector
from network.globsim import SimGlobals as G


class Net(nn.Module):
    """docstring for Net"""

    def __init__(self, **kwargs):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(kwargs['state_count'], 64)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(64, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(32, kwargs['action_count'])
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)
        # action_prob = self.out(x)
        return self.out(x)  # action_prob


class DQN:
    """docstring for DQN"""

    def __init__(self, **kwargs):
        print("{}".format(kwargs))
        super(DQN, self).__init__()

        self.action_count = kwargs['action_count']
        self.state_count = kwargs['state_count']
        net_config = {
            "action_count": self.action_count,
            "state_count": self.state_count,
        }
        self.eval_net, self.target_net = Net(**net_config), Net(**net_config)

        self.mem_capacity = kwargs['mem_capacity']
        self.learning_rate = kwargs['learning_rate']

        self.gamma = kwargs['gamma']

        self.epsilon = kwargs['epsilon']
        self.epsilon_decay = kwargs['epsilon_decay']
        self.epsilon_min = kwargs['epsilon_min']

        self.nn_update = kwargs['nn_update']

        self.batch_size = kwargs['batch_size']

        self.exploration_strategy = kwargs['exploration_strategy']
        self.temperature = float(kwargs['temperature'])

        self.learn_step_counter = 0
        self.step_eps = 0

        self.memory_counter = 0
        self.memory = np.zeros((self.mem_capacity, self.state_count * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate, eps=0.01,
                                          weight_decay=0.0001)
        self.loss_func = nn.MSELoss()

        self.greed_actions = 0
        self.non_greedy_action = 0

        self.fake_episode = 1

        self.stop_learning = False

    def inc_fake_episode(self):
        self.fake_episode += 1
        if self.fake_episode % 1000 == 0:
            self.temperature = max(0.000001, self.temperature * self.epsilon_decay)

    def epsilon_greedy_strategy(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # get a 1D array

        if np.random.random(1)[0] > self.epsilon:  # greedy policy
            self.greed_actions += 1
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
            return action, action_value.squeeze(0).detach().numpy()
        else:  # random policy
            self.non_greedy_action += 1
            action = np.random.randint(0, self.action_count)
            action = action
            return action, None

    def softmax_strategy(self, state):
        self.inc_fake_episode()
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        q_values = self.eval_net.forward(state)
        q_values = q_values.squeeze(0)

        normalised_q = q_values - torch.max(q_values)
        probs = torch.softmax(normalised_q / self.temperature, -1)

        # log_probs = torch.log(probs)
        # entropy = torch.sum(-probs * log_probs)

        action_idx = np.random.choice(self.action_count, p=probs.detach().numpy())

        return action_idx, q_values.detach().numpy()

    def choose_action(self, state):
        self.step_eps += 1

        if self.step_eps % 100 == 0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self.exploration_strategy == "softmax":
            return self.softmax_strategy(state)
        elif self.exploration_strategy == "epsilon_greedy":
            return self.epsilon_greedy_strategy(state)

        # q_vals = self.eval_net.forward(state).squeeze(0)
        # temp = self.get_tempurature()
        # q_temp = q_vals / temp
        # probs = torch.softmax(q_temp - torch.max(q_temp), -1)
        # action_idx = np.random.choice(len(q_temp), p=probs.numpy())

    def store_transition(self, state, action, reward, next_state):


        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.mem_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def stop(self):
        self.stop_learning = True

    def learn(self):



        # update the parameters
        if self.step_eps % self.nn_update == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch from memory
        sample_index = np.random.choice(min(self.mem_capacity, len(self.memory)), self.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.state_count])
        batch_action = torch.LongTensor(batch_memory[:, self.state_count:self.state_count + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.state_count + 1:self.state_count + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.state_count:])

        # q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def run(**run_config):
    print("[+] Starting a new run with the following configurations:")
    print("{}".format(run_config))
    print()
    G.reset()
    G.RESOURCES_COUNT = run_config["network_resources_count"]
    G.INIT_LEARNING_RESOURCES = run_config["learning_resources_count"]

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

    max_steps = 600000
    episodes = 600
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
    for i in range(episodes):

        accumulated_reward = 0
        queue_learner = []
        dqn.greed_actions = 0
        dqn.non_greedy_action = 0

        greedySelector.greedy = 0
        greedySelector.non_greedy = 0
        q = None
        q_count = 0
        for j in range(steps_per_episode):
            step += 1

            ql_size = len(learning_queue)
            queue_learner.append(ql_size)

            action, q_vals = dqn.choose_action(state)

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
                                                                        int(info["interruption"][0][1]),
                                                                        int(info["interruption"][1][1])),
                                                                    DP=(int(state[2]), int(state[3])),
                                                                    E=(info['gp'][0], info['gp'][1]))
            else:
                apply = False

            if apply:
                next_state, reward, done, info = env.step(3, greedy_selection)
            else:
                next_state, reward, done, info = env.step(action)

            all_generated_samples += 1

            new_drp_rate = len(learning_queue) / 1500.0


            # if drop_rate <= np.random.random(1)[0] and not apply:
            if run_config["use_prob_selection"] and new_drp_rate <= np.random.random(1)[0] and not apply:
                learning_queue.push((state, action, reward, next_state))
            elif not run_config["use_prob_selection"]:
                learning_queue.push((state, action, reward, next_state))

            if apply:
                samples = learning_queue.step(additional_resources=greedy_selection[2])
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
                    forwarded_samples += len(samples)
                    for (si, a, r, sj) in samples:
                        dqn.store_transition(si, a, r, sj)

                        # changed from run_config to a fixed thing
                        if dqn.memory_counter >= run_config["batch_size"] * 15:
                            dqn.learn()

                # if done:
                #    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            state = next_state
            if step % 10 == 0:
                wandb.log(
                    {
                        "slice0:queue-size": state[2],
                        "slice0:resources": state[4],
                        "slice0:incoming-traffic": info['gp'][0],
                        "slice0:packet-drop": info['s0'][1],
                        "slice0:packet-dead": info['s0'][3],
                        "slice0:urgent-packet-count": info["interruption"][0][1],

                        "slice1:queue-size": state[3],
                        "slice1:resources": state[5],
                        "slice1:incoming-traffic": info['gp'][1],
                        "slice1:packet-drop": info['s1'][1],
                        "slice1:packet-dead": info['s1'][3],
                        "slice1:urgent-packet-count": info["interruption"][1][1],

                        "learner:queue-size": ql_size,
                        "learner:resources": learning_queue.last_resource_usage,
                        "learner:forwarded-experiences": forwarded_samples,
                        "learner:greedySelectionEPS": greedySelector.g_eps

                    },
                    step=step
                )

            # plotter.push_data(q1=state[2], q2=state[3], r1=state[4], r2=state[5],
            #                  gp1=info['gp'][0], gp2=info['gp'][1])

            accumulated_reward += reward

        # print("greedy: {} -- non-greedy: {}".format(dqn.greed_actions, dqn.non_greedy_action))
        # plotter.push_episodic_stats(accumulated_reward, dqn.epsilon,
        #                            average_q_learner=forwarded_samples / float(all_generated_samples))

        # plotter.plot()

        print("[+] Greedy: {} - ({}) -- Non-Greedy: {} - ({})".format(greedySelector.greedy, env.reward_greedy,
                                                                      greedySelector.non_greedy, env.reward_non_greedy))
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
    max_users_conf = [(16, 16)]
    init_resources_conf = [

        (14, 1, 0.0)

    ]
    learning_rates = []
    gammas = []
    nn_updates = []
    batch_sizes = []
    network_resources_counts = []
    strategy = "epsilon_greedy"
    for (u1, u2) in max_users_conf:
        for (nres, lres, drp) in init_resources_conf:
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            config = {
                "use_prob_selection": False,
                "use_greedy": False,
                "queue_type": "fifo",
                "g_eps": 3.0 / 15.0,
                "g_eps_decay": 0.99998849492,
                "g_eps_min": 0.00001,

                "mem_capacity": 10000,
                "gamma": 0.95,
                "epsilon": 1.0,
                "epsilon_decay": 0.9995,
                "epsilon_min": 0.0001,
                "nn_update": 50,
                "batch_size": 32,
                "learning_rate": 0.00001,
                "temperature": 1.2,
                "exploration_strategy": strategy,

                "network_resources_count": nres,
                "learning_resources_count": lres,

                "max_users:0": u1,
                "max_users:1": u2,

                "drop_rate": drp,

                "slice0:P00": G.P00,
                "slice0:P01": G.P01,
                "slice0:P11": G.P11,
                "slice0:P10": G.P10,

                "slice1:P00": G.P00,
                "slice1:P01": G.P01,
                "slice1:P11": G.P11,
                "slice1:P10": G.P10,
            }
            wandb.init(reinit=True, config=config, project="costoflearning-ts=0.001")
            wandb.run.name = "sim-res={}/{},u={}/{},drp={},useG={}".format(nres, lres, u1, u2, drp, config["use_greedy"])
            wandb.run.save()

            run(**config)

            wandb.finish()


run_experiments()