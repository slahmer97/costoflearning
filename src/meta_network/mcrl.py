import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from src.meta_network.exploration_strategy import EpsilonGreedyPolicy, SoftmaxPolicy


class Net(nn.Module):
    """docstring for Net"""

    def __init__(self, **kwargs):
        super(Net, self).__init__()
        self.batch = kwargs["batch_size"]
        # self.disc = kwargs["discreet_ctx_count"]
        self.disc_rep = kwargs["discrete_rep_size"]
        self.contc = kwargs["continous_ctx_count"]
        # self.max_embedding_index = kwargs["max_embedding_index"]
        self.env_state_size = kwargs['state_count']
        # self.ctx_encoder1 = nn.Embedding(self.max_embedding_index, self.disc_rep, max_norm=True)

        self.ctx_encoder2 = nn.Linear(self.contc + self.disc_rep, 8)
        self.ctx_encoder2.weight.data.normal_(0, 0.1)

        self.fc1 = nn.Linear(6, 64)
        self.fc1.weight.data.normal_(0, 0.1)

        self.fc2 = nn.Linear(64 + 8, 96)
        self.fc2.weight.data.normal_(0, 0.1)

        self.fc3 = nn.Linear(96, 64)
        self.fc3.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(64, kwargs['action_count'])
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        batch = x.shape[0]

        slice1 = x[:, :self.contc + self.disc_rep].view(batch, self.contc + self.disc_rep)
        slice3 = x[:, self.contc + self.disc_rep:].view(batch, 6)

        slice1 = F.relu(self.ctx_encoder2(slice1))

        slice3 = F.relu(self.fc1(slice3))
        x = torch.cat((slice1, slice3), 1)
        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        # action_prob = self.out(x)
        return self.out(x)  # action_prob

    def save_me(self, name="eval"):
        torch.save(self.state_dict(), "{}-model.pt".format(name))

    def load_me(self, name="eval"):
        self.load_state_dict(torch.load("{}-model.pt".format(name)))


class DQN:
    """docstring for DQN"""

    def __init__(self, **kwargs):
        print("{}".format(kwargs))
        super(DQN, self).__init__()

        self.action_count = kwargs['action_count']
        self.state_count = kwargs['state_count']
        self.setup_exploration_strategy(**kwargs['exploration_strategy'])
        self.eval_net, self.target_net = Net(**kwargs), Net(**kwargs)
        self.learning_rate = kwargs['learning_rate']
        self.gamma = kwargs['gamma']
        self.nn_update = kwargs['nn_update']
        self.batch_size = kwargs['batch_size']
        self.exploration_strategy = kwargs['exploration_strategy']
        self.learn_step_counter = 0
        self.step_eps = 0

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate, eps=0.01,
                                          weight_decay=0.0001)

        self.loss_func = nn.MSELoss()

        self.stop_learning = False

    def setup_exploration_strategy(self, **strategy_kwargs):
        policy_name = strategy_kwargs['name']
        if policy_name == "epsilon_greedy":
            print("here")
            self.policy = EpsilonGreedyPolicy(self.action_count, **strategy_kwargs)
        elif policy_name == "softmax":
            self.policy = SoftmaxPolicy(self.action_count, **strategy_kwargs)
        elif policy_name == "ucb":
            #self.policy = UCBPolicy(self.action_count, **strategy_kwargs)
            raise Exception("TODO exploration strategy")

        else:
            raise Exception("Unknown exploration strategy")

    def choose_action(self, state):
        self.step_eps += 1

        return self.policy.select_action(state, self.eval_net)

    def reset_epsilon(self, start=1.0, end=100):
        self.start = start
        self.epsilon = start
        # self.slope = np.log(self.epsilon_min / self.start) / end
        self.i = 0
        # print("nez eps: {} -- slope: {} -- start: {} -- end: {}".format(self.epsilon, self.slope, start, end))

    def compute_td_error(self, state, action, reward, next_state):
        state = torch.FloatTensor(state).view(1, -1)
        actions = torch.LongTensor([action]).view(1, 1)
        reward = torch.FloatTensor([reward]).view(1, 1)
        next_state = torch.FloatTensor(next_state).view(1, -1)

        state_action_values = self.eval_net.forward(state)
        state_value = state_action_values.gather(1, actions)

        next_state_values = self.target_net.forward(next_state).detach()
        next_state_value = next_state_values.max(1)[0]

        expected_state_action_value = reward + self.gamma * next_state_value

        td_error = expected_state_action_value - state_value
        return torch.abs(td_error.detach()).item()

    def stop(self):
        self.stop_learning = True

    def learn(self, memory):
        # update the parameters
        if self.step_eps % self.nn_update == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        batch_state, batch_action, batch_reward, batch_next_state = memory.sample()
        # sample batch from memory

        # q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)
        value = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return value
