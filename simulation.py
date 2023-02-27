from meta_network.network import MetaNetwork
import numpy as np


class RandomPolicy:
    def choose_action(self, state):
        return np.random.randint(0, 3), None, None


class Simulation:

    def __init__(self, **cfg):
        self._cfg = cfg
        self.Transitions = cfg.get("transitions")
        self.RESOURCES_COUNT = cfg.get("resources_count")
        self._cost_weights = cfg.get("cost_weights")
        self.NET_TIMESLOT_STEP = 0
        self.flow_counter = 0
        self.NET_TIMESLOT_DURATION_S = 0.001  # in s
        self.BANDWIDTH_PER_RESOURCE = 513000  # byte per second // 0.1 mbps
        self.TOTAL_LINK_BANDWIDTH = self.RESOURCES_COUNT * self.BANDWIDTH_PER_RESOURCE

        self.EXPERIENCE_SIZE = 1500  # in bits

        self.last_state = None

        self.ctx = np.array([self._cfg["max_users:0"], self._cfg["max_users:1"], self._cfg["resources_count"],
                             self.Transitions[0][0][0], self.Transitions[0][0][1],
                             self.Transitions[0][1][0], self.Transitions[0][1][1],

                             self.Transitions[1][0][0], self.Transitions[1][0][1],
                             self.Transitions[1][1][0], self.Transitions[1][1][1],

                             ])
        self._env = MetaNetwork(**cfg, sim=self)

    def add_ctx(self, state):
        return np.concatenate((self.ctx, state), axis=0)

    def reset(self):
        self.NET_TIMESLOT_STEP = 0

        self.last_state = self._env.reset()
        self.last_state = self.add_ctx(self.last_state)
        return self.last_state

    def rollout(self, policy, k_steps):
        if self.last_state is None:
            self.reset()

        ret = []
        cum = 0
        for i in range(k_steps):
            action, _, _ = policy.choose_action(self.last_state)
            new_state, reward, done, info = self._env.step(action)
            new_state = self.add_ctx(new_state)
            ret.append((self.last_state, action, reward, new_state, done, info))
            cum += reward[0]
            self.last_state = new_state
        return ret, cum

    def send_success(self):
        return True

    def urllc_cost(self, pkt_latency):
        if pkt_latency < 50:
            return 0.0
        else:
            return - (1.0 / 70.0) * pkt_latency
