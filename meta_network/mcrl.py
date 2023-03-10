import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


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

        self.state_ctx_count = kwargs['state_count'] + kwargs["continous_ctx_count"] + 1
        self.eval_net, self.target_net = Net(**kwargs), Net(**kwargs)

        self.mem_capacity = kwargs['mem_capacity']
        self.learning_rate = kwargs['learning_rate']
        self.learning_rate2 = kwargs['learning_rate2']

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
        self.memory = np.zeros((self.mem_capacity, self.state_ctx_count * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate, eps=0.01,
                                          weight_decay=0.0001)
        self.optimizer2 = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate2, eps=0.01,
                                           weight_decay=0.0001)
        self.loss_func = nn.MSELoss()

        self.greed_actions = 0
        self.non_greedy_action = 0

        self.fake_episode = 1

        self.stop_learning = False

        self.slope = 1
        self.start = 1.0

        self.i = 0
        self.reset_epsilon()

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
            currentStep = int(self.i / 1000)
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

    def reset_epsilon(self, start=1.0, end=100):
        self.start = start
        self.epsilon = start
        self.slope = np.log(self.epsilon_min / self.start) / end
        self.i = 0
        print("nez eps: {} -- slope: {} -- start: {} -- end: {}".format(self.epsilon, self.slope, start, end))

    def reset_mem(self):
        self.memory_counter = 0
        self.memory = np.zeros((self.mem_capacity, self.state_ctx_count * 2 + 2))

    def stop(self):
        self.stop_learning = True

    def learn(self, memory):
        # update the parameters
        if self.step_eps % self.nn_update == 0:
            #print("target")
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

    def meta_learn(self, memory, zero=True):
        batch_state, batch_action, batch_reward, batch_next_state, _ = memory.sample()

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        if zero:
            self.optimizer.zero_grad()
        value = loss.item()
        loss.backward()

        return value