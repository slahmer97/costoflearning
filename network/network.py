from gym.vector.utils import spaces

from network.flow_gen import OnOffFlowGenerator, UniformFlowGenerator, AggregateOnOffFlowGenerator
import numpy as np
import gym
from network.globsim import SimGlobals as G
from network.netqueue import NetQueue


class Network(gym.Env):

    def render(self, mode="human"):
        pass

    def __init__(self, **kwargs):
        self.end = False
        self.state = None
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1500, shape=(6,), dtype=np.float32)
        self.packet_size = 512.0  # bits
        self.slice_0_flows = [
            {
                'max_users': kwargs["max_users:0"],
                "packet_size": self.packet_size,
                "req_delay": 0.300,
                "max_delay": np.infty,
                "on_rate": 1536000,  # bits per second
                "flow_class": 'non-critical-audio',
                "flow_model": 'unicast',
                "flow_performance": 'strict',
                "slice": 0
            }

        ]

        self.slice_1_flows = [
            {
                'max_users': kwargs["max_users:1"],
                "packet_size": self.packet_size,
                "req_delay": 0.0705,
                "max_delay": 0.0705,
                "on_rate": 1536000,  # bits per second
                "flow_class": 'critical-audio',
                "flow_model": 'unicast',
                "flow_performance": 'strict',
                "slice": 1
            }
        ]
        self.normaliser = np.array(
            [float(kwargs["max_users:0"]), float(kwargs["max_users:1"]), 1500.0, 1500.0, G.RESOURCES_COUNT,
             G.RESOURCES_COUNT])
        self.action0 = 0
        self.action1 = 0
        self.action2 = 0
        self.slices = []
        for i in range(2):
            self.slices.append(NetQueue())

        self.init_resources()

        self.generators = []

        for conf in self.slice_0_flows:
            self.generators.append(AggregateOnOffFlowGenerator(**conf))

        for conf in self.slice_1_flows:
            self.generators.append(AggregateOnOffFlowGenerator(**conf))

        r1, r2 = self.get_expected_rates()  # packets per second
        total_number_of_packets = G.TOTAL_LINK_BANDWIDTH / self.packet_size
        self.total_number_of_packets_per_slot = total_number_of_packets * G.NET_TIMESLOT_DURATION_S
        print("[+] Total available bandwidth: {} p/s -- {} p/slot".format(total_number_of_packets,
                                                                          self.total_number_of_packets_per_slot))
        print("[+] Expected number of packets:\n"
              "\t slice1 : {} packet/sec -- {} packet/slot\n"
              "\t slice2 : {} packet/sec -- {} packet/slot\n".format(r1, r1 * G.NET_TIMESLOT_DURATION_S,
                                                                     r2, r2 * G.NET_TIMESLOT_DURATION_S))

    def init_resources(self):
        res = np.random.randint(G.RESOURCES_COUNT + 1, size=1)[0]
        self.slices[0].allocate_resource(res)
        self.slices[1].allocate_resource(G.RESOURCES_COUNT - res)

    def step(self, action=None):

        def reward_func3(obj, q1, dead1, drop1, q2, dead2, drop2):
            normalized_drop1 = drop1 / obj.slice_0_flows[0]["max_users"]

            normalized_dead2 = dead2 / obj.slice_1_flows[0]["max_users"]
            return - normalized_drop1 - normalized_dead2

        def reward_func2(q1, d1, q2, d2):
            normalized_q1 = q1 / 1500.0
            normalized_d1 = d1

            normalized_q2 = q2 / 1500.0
            normalized_d2 = np.exp(d2 / 1500.0)

            normalized_diff = abs((q1 - q2) / 1500.0)

            return - normalized_q1 - np.exp(normalized_q2) - normalized_diff

        def reward_func(q1, q2):
            normalized_q1 = q1 / 1500.0
            normalized_q2 = q2 / 1500.0
            normalized_diff = abs((q1 - q2) / 1500.0)
            term1 = - pow(q1, 1) / 1500.0
            term2 = - pow(q2, 1) / 1500.0
            return - normalized_q1 - normalized_q2 - normalized_diff

        tmp_state = []
        stats = []
        served_packets = []

        for generator in self.generators:
            # generate the packets before running the activeuser model to the next state, then makes the move
            a, active_user, _ = generator.step()
            self.slices[generator.slice].enqueue(a)
            tmp_state.append(active_user)
            stats.append(len(a))
        # Apply the new resource allocation according to the action=action
        if action == 0:
            self.action0 += 1
            pass
        elif action == 1:
            self.action1 += 1
            self.slices[0].allocate_resource(min(G.RESOURCES_COUNT, self.slices[0].allocated_resources + 1))
            self.slices[1].allocate_resource(G.RESOURCES_COUNT - self.slices[0].allocated_resources)

        elif action == 2:
            self.action2 += 1

            # move one resource to slice 1
            self.slices[0].allocate_resource(max(0, self.slices[0].allocated_resources - 1))
            self.slices[1].allocate_resource(G.RESOURCES_COUNT - self.slices[0].allocated_resources)
        else:
            # action was not recognized
            raise ValueError("Action value was not recognized")

        # move the system
        s = []  # [s0, s1]
        for i in range(2):
            served = self.slices[i].step()
            served_packets.append(served)

            self.slices[i].update_dead_packets()
            s.append(self.slices[i].get_state())

            self.slices[i].reset_temp_stats()
            tmp_state.append(len(self.slices[i]))

        G.NET_TIMESLOT_STEP += 1

        self.state = tmp_state
        self.state.append(self.slices[0].allocated_resources)
        self.state.append(self.slices[1].allocated_resources)

        info = {
            "episode": 0,
            "gp": stats,
            "served_packets": served_packets,
            "s0": s[0],
            "s1": s[1],
        }

        done = self.end
        if self.end:
            self.end = False

        ret_reward = reward_func3(obj=self, q1=self.state[2], dead1=s[0][3], drop1=s[0][1], q2=self.state[3],
                                  dead2=s[1][3], drop2=s[1][1])
        # ret_reward = reward_func2(self.state[2], s[0][3], self.state[3], s[1][3])
        # ret_reward = reward_func(self.state[2], self.state[3])

        return np.array(self.state) / self.normaliser, ret_reward, done, info

    def reset(self, **kwargs):
        self.action0 = 0
        self.action1 = 0
        self.action2 = 0
        lambda_1, _ = self.generators[0].reset()
        lambda_2, _ = self.generators[1].reset()
        self.slices = []
        for i in range(2):
            self.slices.append(NetQueue())
        self.init_resources()
        self.state = [lambda_1, lambda_2, len(self.slices[0]), len(self.slices[1]), self.slices[0].allocated_resources,
                      self.slices[1].allocated_resources]
        return np.array(self.state) / self.normaliser

    def set_end(self):
        self.end = True

    def get_expected_rates(self):
        p_on = G.P01 / (G.P01 + G.P10)
        lambda0 = self.slice_0_flows[0]["max_users"] * self.slice_0_flows[0]["on_rate"] * p_on
        lambda1 = self.slice_1_flows[0]["max_users"] * self.slice_1_flows[0]["on_rate"] * p_on
        return lambda0 / 150.0, lambda1 / 150.0
