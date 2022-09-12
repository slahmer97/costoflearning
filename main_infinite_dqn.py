import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network.network import Network
from collections import deque
import matplotlib.pyplot as plt
import copy

# hyper-parameters
BATCH_SIZE = 128
LR = 0.001
GAMMA = 0.95
EPS = 1.0
EPS_DECAY = 0.999
EPS_MIN = 0.001
MEMORY_CAPACITY = 10000
Q_NETWORK_ITERATION = 100
env = net = Network()
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape


class Net(nn.Module):
    """docstring for Net"""

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50, 30)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, NUM_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob


class DQN():
    """docstring for DQN"""

    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.step_eps = 0

        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        global EPS, EPS_MIN, EPS_DECAY
        self.step_eps += 1

        if self.step_eps % 100 == 0:
            EPS = max(EPS_MIN, EPS * EPS_DECAY)

        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # get a 1D array
        if np.random.randn() > EPS:  # greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:  # random policy
            action = np.random.randint(0, NUM_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):

        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES + 1:NUM_STATES + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:])

        # q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():
    dqn = DQN()
    episodes = 1000000
    print("Collecting Experience....")

    plt.ion()
    fig, axs = plt.subplots(4)
    mean_reward = []

    for i in range(episodes):
        # print("episode: {}".format(i))
        state = env.reset()
        max_plot = 300
        epsilons = deque(maxlen=max_plot)

        queue1 = deque(maxlen=max_plot)
        queue1.append(state[2])
        queue2 = deque(maxlen=max_plot)
        queue2.append(state[3])

        reward_list = []

        resources1 = deque(maxlen=max_plot)
        resources1.append(state[4])
        resources2 = deque(maxlen=max_plot)
        resources2.append(state[5])

        r = 0
        for _ in range(1000):

            action = dqn.choose_action(state)
            next_state, reward, done, info = env.step(action)

            dqn.store_transition(state, action, reward, next_state)
            epsilons.append(EPS)
            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
                # if done:
                #    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))

            state = next_state

            queue1.append(state[2])
            queue2.append(state[3])

            resources1.append(state[4])
            resources2.append(state[5])

            r += reward

        reward_list.append(r)

        epsilons.append(EPS)

        mean_reward.append(reward_list[-min(100, len(reward_list)):])
        axs[0].cla()
        axs[0].plot(mean_reward)
        axs[0].set_title('Average Sum of Reward over 1000 step')

        axs[1].cla()
        axs[1].plot(list(queue1), label='q1')
        axs[1].plot(list(queue2), label='q2')
        #axs[1].set_title('Queue Size -- ')

        axs[2].cla()
        axs[2].plot(list(epsilons), label='EPS')
        # plt.legend()

        axs[3].cla()
        axs[3].plot(list(resources1), label='resource-1')
        axs[3].plot(list(resources2), label='resource-2')
        plt.pause(0.0001)


if __name__ == '__main__':
    main()
