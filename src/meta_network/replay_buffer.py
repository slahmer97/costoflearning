import numpy as np
import torch

TensorType = torch.Tensor


class ReplayBuffer0:
    # self, env_obs_size, capacity, batch_size, device
    def __init__(self, input_shape, max_size, batch_size):
        self.batch_size = batch_size
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, 1))
        self.reward_memory = np.zeros((self.mem_size, 1))

    def add(self, state, action, reward, state_):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_

        self.mem_cntr += 1

    def sample(self, batch_size=-1):
        max_mem = min(self.mem_cntr, self.mem_size)
        if batch_size <= 0:
            bs = self.batch_size
        else:
            bs = batch_size

        batch = np.random.choice(max_mem, bs)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]

        states = torch.FloatTensor(states).view(bs, -1)
        actions = torch.LongTensor(actions).view(bs, 1)
        rewards = torch.FloatTensor(rewards).view(bs, 1)
        states_ = torch.FloatTensor(states_).view(bs, -1)

        return states, actions, rewards, states_

    def is_sufficient(self):
        return self.mem_cntr >= self.batch_size * 10


class ReplayBuffer(object):

    def __init__(
            self, env_obs_size, capacity, batch_size, device
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        self.env_obses = np.empty((capacity, env_obs_size), dtype=np.float32)
        self.next_env_obses = np.empty((capacity, env_obs_size), dtype=np.float32)
        self.actions = np.empty((capacity, 1), dtype=np.int)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def is_empty(self):
        return self.idx == 0

    def add(self, env_obs, action, reward, next_env_obs):
        np.copyto(self.env_obses[self.idx], env_obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_env_obses[self.idx], next_env_obs)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == (self.capacity - 1)

    def sample(self, index=None):
        if index is None:
            idxs = np.random.randint(
                0, self.capacity if self.full else self.idx, size=self.batch_size
            )
        else:
            idxs = index

        env_obses = torch.as_tensor(self.env_obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_env_obses = torch.as_tensor(
            self.next_env_obses[idxs], device=self.device
        ).float()

        return (
            env_obses, actions, rewards, next_env_obses)

    def reset(self):
        self.idx = 0

    def is_sufficient(self):
        return self.idx >= self.batch_size * 10
