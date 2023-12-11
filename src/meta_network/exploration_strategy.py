import numpy as np
import torch


class ExplorationPolicy:
    def __init__(self, action_count, **kwargs):
        self.action_count = action_count

    def select_action(self, state, network):
        raise NotImplementedError

    def set_end(self):
        raise NotImplementedError


class EpsilonGreedyPolicy(ExplorationPolicy):
    def __init__(self, action_count, **kwargs):
        super().__init__(action_count)
        self.epsilon = kwargs['epsilon']
        self.epsilon_decay = kwargs['epsilon_decay']
        self.epsilon_min = kwargs['epsilon_min']

        self.counter = 0
        self.update = kwargs['update']

    def set_end(self):
        self.epsilon = self.epsilon_min

    def select_action(self, state, network):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        self.counter += 1
        if self.counter % self.update:
            self.update_epsilon()
        if np.random.random(1)[0] > self.epsilon:
            action_value = network.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
            return action, 0, action_value.squeeze(0).detach().numpy()
        else:  # random policy
            action = np.random.randint(0, self.action_count)
            action = action
            return action, 1, None

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


class SoftmaxPolicy(ExplorationPolicy):
    def __init__(self, action_count, **kwargs):
        super().__init__(action_count)
        self.temperature = kwargs['temperature']

    def select_action(self, state, network=None):
        state_tensor = torch.unsqueeze(torch.FloatTensor(state), 0)
        action_values = network.forward(state_tensor).detach().numpy()
        probabilities = np.exp(action_values / self.temperature) / np.sum(np.exp(action_values / self.temperature))
        return np.random.choice(range(self.action_count), p=probabilities.ravel())


