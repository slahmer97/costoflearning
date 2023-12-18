import numpy as np
from collections import deque


def get_policy_selector(**kwargs):
    policy_name = kwargs.get("name")
    if policy_name == "simple-selector":
        return SimpleSelector(**kwargs)
    elif policy_name == "dynamic-selector":
        return DynamicSelector(**kwargs)
    else:
        raise ValueError("Trying to use unknown selector policy")


class PolicySelector:
    def __init__(self, **kwargs):
        pass

    def choose_action(self, **kwargs):
        raise NotImplementedError

    def reset(self):
        pass

    def push_stats(self, **kwargs):
        pass

    def set_end(self):
        raise NotImplementedError


import csv


class DynamicSelector(PolicySelector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.rho = kwargs.get("rho_init", 0.2)
        self.rho_init = self.rho
        self.rho_max = kwargs.get("rho_max", 0.22)
        self.rho_min = kwargs.get("rho_min", 0.001)
        self.gamma = kwargs.get("gamma", 0.95)

        self.window_size = kwargs.get("ma_tde_ws", 100)
        self.td_errors_deque = deque(maxlen=self.window_size)
        self.stde_deque = deque(maxlen=kwargs.get("stde_ws", 200))
        self.last_action = None
        self.counter = 0
        self.update = 100
        self.file = open(f"data/rho_evo-task-{kwargs.get('task_id')}.csv", "w")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["derivative", "rho"])

    def choose_action(self, **kwargs):
        if np.random.random() < self.rho:
            self.last_action = 2  # Greedy
        else:
            self.last_action = 1  # RL
        return self.last_action

    def push_stats(self, **kwargs):
        td_error = kwargs.get('td_error')
        self.td_errors_deque.append(td_error)
        stde = np.mean(self.td_errors_deque)
        self.stde_deque.append(stde)

        self.counter += 1

        if self.counter % self.update == 0:
            self.adjust_rho()

    def adjust_rho(self):
        derivative = 0
        if len(self.stde_deque) >= 102:
            # Compute the derivative of STDE
            derivative = self.stde_deque[-1] - self.stde_deque[-101]
            #print(f"Derivative: {derivative}")
            # Adjust rho based on the derivative of STDE
            if derivative <= 0.001:
                self.rho = max(self.rho * self.gamma, self.rho_min)
            else:
                self.rho = min(self.rho / self.gamma, self.rho_max)
        self.writer.writerow([derivative, self.rho])
        self.file.flush()


class SimpleSelector(PolicySelector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rho_init = kwargs.get("rho_init", 0.2)
        self.rho = self.rho_init
        self.gamma = kwargs.get("gamma", 0.99)
        self.rho_max = kwargs.get("rho_max", 0.22)
        self.rho_min = kwargs.get("rho_min", 0.001)
        self.use = True
        self.reset()
        self.step = 0

    def set_end(self):
        self.rho = self.rho_min

    def reset(self, **kwargs):
        self.rho = kwargs.get("rho_init", self.rho_init)

    def choose_action(self, **kwargs):
        if not self.use:
            return 1

        self.step += 1

        if self.step % 1000 == 0:
            self.rho = max(self.rho * self.gamma, self.rho_min)

        if np.random.random(1)[0] > self.rho or self.step <= 1:
            return 1
        else:
            return 2
