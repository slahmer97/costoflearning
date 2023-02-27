from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch

TensorType = torch.Tensor


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
        self.not_dones = np.empty((capacity, 1), dtype=int)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def is_empty(self):
        return self.idx == 0

    def add(self, env_obs, action, reward, next_env_obs, done=False):
        np.copyto(self.env_obses[self.idx], env_obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_env_obses[self.idx], next_env_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

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
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return (
            env_obses, actions, rewards, next_env_obses, not_dones)

    def reset(self):
        self.idx = 0
