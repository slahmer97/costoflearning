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


class DynamicSelector(PolicySelector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rho = kwargs.get("rho_init", 0.2)
        self.rho_max = kwargs.get("rho_max", 0.22)
        self.rho_min = kwargs.get("rho_min", 0.001)
        self.increase_threshold = kwargs.get("increase_threshold", 0.001)
        self.decrease_threshold = kwargs.get("decrease_threshold", -0.001)

        self.initial_threshold = kwargs.get("initial_threshold", 0.005)

        self.gamma = 0.97
        self.td_error_increase_rate = kwargs.get("td_error_increase_rate", 0.001)
        self.td_error_decrease_rate = kwargs.get("td_error_decrease_rate", 0.01)
        self.count = 0
        self.cumulative_diff = 0.0
        self.last_avg_td_error = None
        self.last_avg_q1 = None
        self.last_avg_q2 = None
        self.last_avg_ql = None

        self.avg_td_error_hist = deque(maxlen=1000)
        self.avg_q1_hist = deque(maxlen=100)
        self.avg_q2_hist = deque(maxlen=100)
        self.avg_ql_hist = deque(maxlen=100)

    def set_end(self):
        self.rho = self.rho_min

    def choose_action(self, **kwargs):
        if np.random.random() < self.rho:
            self.last_action = 2  # Greedy
        else:
            self.last_action = 1  # RL
        return self.last_action

    def push_stats(self, **kwargs):
        self.count += 1
        self.avg_td_error_hist.append(kwargs.get('td_error'))
        if self.count % 750 == 0:
            self._update_rho()

    def _is_first(self):
        return self.last_avg_td_error is None

    def _detect_trend(self, new_avg):

        immediate_diff = new_avg - self.last_avg_td_error
        self.cumulative_diff += immediate_diff

        if immediate_diff > self.initial_threshold:
            self.cumulative_diff = 0
            return 'increasing', immediate_diff
        elif immediate_diff < -self.initial_threshold:
            self.cumulative_diff = 0
            return 'decreasing', immediate_diff
        else:
            return 'stable', immediate_diff

    def _update_rho(self):
        new_avg_td_error = np.mean(self.avg_td_error_hist)

        if self.last_avg_td_error is not None:
            td_error_trend, imm_diff = self._detect_trend(new_avg_td_error)

            if td_error_trend == 'increasing':
                self.rho = min(self.rho / self.gamma, self.rho_max)
            elif td_error_trend == 'decreasing' or td_error_trend == "stable":
                self.rho = max(self.rho * self.gamma, self.rho_min)

        self.last_avg_td_error = new_avg_td_error


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
