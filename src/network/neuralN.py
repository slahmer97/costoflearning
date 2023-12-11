import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


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

    def save_me(self, name="eval"):
        torch.save(self.state_dict(), "{}-ofb.pt".format(name))

    def load_me(self, name="eval"):
        self.load_state_dict(torch.load("{}-ofb.pt".format(name)))


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


        self.slope = 1
        self.start = 1.0

        self.i = 0

    def epsilon_greedy_strategy(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # get a 1D array

        if np.random.random(1)[0] > self.epsilon:  # greedy policy
            self.greed_actions += 1
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
            return action, 0, action_value.squeeze(0).detach().numpy()
        else:  # random policy
            self.non_greedy_action += 1
            action = np.random.randint(0, self.action_count)
            action = action
            return action, 1, None

    def softmax_strategy(self, state):
        return None, None

    def choose_action(self, state):
        self.step_eps += 1

        if self.i % 1000 == 0:
            currentStep =  int(self.i / 1000)
            self.epsilon = self.start * np.exp(self.slope * currentStep)
            self.epsilon = max(self.epsilon_min, self.epsilon)

        self.i += 1
        if self.exploration_strategy == "epsilon_greedy":
            return self.epsilon_greedy_strategy(state)
        else:
            raise Exception("unknown exploration strategy")

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

    def reset_epsilon(self, start=0.3, end=100):
        self.start = start
        self.epsilon = start
        self.slope = np.log(self.epsilon_min / self.start) / end
        self.i = 0
        print("nez eps: {} -- slope: {} -- start: {} -- end: {}".format(self.epsilon, self.slope, start, end))

    def reset_mem(self):
        self.memory_counter = 0
        self.memory = np.zeros((self.mem_capacity, self.state_count * 2 + 2))

    def stop(self):
        self.stop_learning = True

    def learn(self):

        # update the parameters
        if self.step_eps % self.nn_update == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch from memory
        sample_index = np.random.choice(min(self.memory_counter, self.mem_capacity), self.batch_size)
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
