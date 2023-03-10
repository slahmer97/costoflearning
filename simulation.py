from meta_network.network import MetaNetwork
import numpy as np

from collections import deque
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
        self.last_greedy_state = None
        self.ctx = np.array([self._cfg["max_users:0"], self._cfg["max_users:1"], self._cfg["resources_count"],
                             self.Transitions[0][0][0], self.Transitions[0][0][1],
                             self.Transitions[0][1][0], self.Transitions[0][1][1],

                             self.Transitions[1][0][0], self.Transitions[1][0][1],
                             self.Transitions[1][1][0], self.Transitions[1][1][1],

                             ])
        self._env = MetaNetwork(**cfg, sim=self)

        self.greedy_counter = 0
        self.non_greedy_counter = 0
        self.averge_queue0 = deque(maxlen=300)
        self.averge_queue0.append(0)
        self.averge_queue1 = deque(maxlen=300)
        self.averge_queue1.append(0)

    def add_ctx(self, state):
        return np.concatenate((self.ctx, state), axis=0)

    def reset(self):
        self.NET_TIMESLOT_STEP = 0

        self.last_state = self._env.reset()
        self.last_state = self.add_ctx(self.last_state)
        return self.last_state

    def reset_greedy_counters(self):
        self.greedy_counter = 0
        self.non_greedy_counter = 0
        self._env.reward_greedy = 0
        self._env.reward_non_greedy = 0


    def get_greedy_info(self):
        return self.greedy_counter, self.non_greedy_counter, self._env.reward_greedy, self._env.reward_non_greedy

    def rollout(self, dnn_policy, greedy_policy, policy_selector, k_steps, meta_data):
        if self.last_state is None:
            self.reset()

        ret = []
        cum = 0
        learning_resources = 0
        for i in range(k_steps):

            p = policy_selector.choose_action()
            if p == 1:
                action, _, _ = dnn_policy.choose_action(self.last_state)
                new_state, reward, done, info = self._env.step(action)
                self.non_greedy_counter += 1
            elif p == 2:
                slice0_qsize, slice1_qsize, slice1_up, slice1sp, learning_pp = self.last_greedy_state
                r0, r1, r2 = greedy_policy.choose_action(slice0_qsize, slice1_qsize, slice1_up, slice1sp, learning_pp, np.mean(self.averge_queue0), np.mean(self.averge_queue1))
                action = 3
                new_state, reward, done, info = self._env.step(action, (r0, r1))
                learning_resources += r2
                self.greedy_counter += 1
            else:
                raise Exception('Unknown p value')
            # slice0_qsize, slice1_up, slice1sp, learning_pp
            self.averge_queue0.append(info["queue_size"][0])
            self.averge_queue1.append(info["queue_size"][1])

            self.last_greedy_state = (
                info["queue_size"][0] + info["active_users"][0], info["queue_size"][1] + info["active_users"][1],
                info["packet_urgent"][0], info["packet_urgent"][1], len(meta_data["cp"]))

            new_state = self.add_ctx(new_state)
            ret.append((self.last_state, action, reward, new_state, done, info))
            cum += reward[0]
            self.last_state = new_state

        return ret, cum, learning_resources

    def send_success(self):
        return True

    def urllc_cost(self, pkt_latency):
        if pkt_latency < 50:
            return 0.0
        else:
            return - (1.0 / 70.0) * pkt_latency
