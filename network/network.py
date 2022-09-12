from typing import Optional, Union, List

from gym.vector.utils import spaces

from network.flow_gen import OnOffFlowGenerator, UniformFlowGenerator, AggregateOnOffFlowGenerator
import numpy as np
import gym
from network.globsim import SimGlobals
from network.netqueue import NetQueue


class Network(gym.Env):

    def render(self, mode="human"):
        pass

    def __init__(self, **kwargs):
        self.end = False
        self.state = None
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1500, shape=(6,), dtype=np.float32)
        self.slice_0_flows = [
            {
                'max_users': 8,
                "packet_size": 512,
                "req_delay": 300,
                "max_delay": np.infty,
                "on_rate": 1536000,
                "flow_class": 'non-critical-audio',
                "flow_model": 'unicast',
                "flow_performance": 'strict',
                "slice": 0
            }

        ]

        self.slice_1_flows = [
            {
                'max_users': 9,
                "packet_size": 512,
                "req_delay": 70.5,
                "max_delay": 70.5,
                "on_rate": 1536000,
                "flow_class": 'critical-audio',
                "flow_model": 'unicast',
                "flow_performance": 'strict',
                "slice": 1
            }
        ]
        self.action0 = 0
        self.action1 = 0
        self.action2 = 0
        self.slices = []
        for i in range(2):
            self.slices.append(NetQueue())

        res = 5
        self.slices[0].allocate_resource(res)
        self.slices[1].allocate_resource(10 - res)

        self.generators = []

        for conf in self.slice_0_flows:
            self.generators.append(AggregateOnOffFlowGenerator(**conf))

        for conf in self.slice_1_flows:
            self.generators.append(AggregateOnOffFlowGenerator(**conf))

    def step(self, action=None):
        def reward_func(q1, q2):
            normalized_q1 = q1 / 1500.0
            normalized_q2 = q2 / 1500.0
            normalized_diff = abs((q1 - q2) / 1500.0)
            term1 = - pow(q1, 1) / 1500.0
            term2 = - pow(q2, 1) / 1500.0
            return - normalized_q1 - normalized_q2 - normalized_diff

        tmp_state = []
        stats = []
        for generator in self.generators:
            # generate the packets before running the activeuser model to the next state, then makes the move
            a, active_user, _ = generator.step()
            self.slices[generator.slice].enqueue(a)
            tmp_state.append(active_user)
            stats.append(len(a) * 512)
        # Apply the new resource allocation according to the action=action
        if action == 0:
            self.action0 += 1
            pass
        elif action == 1:
            self.action1 += 1
            self.slices[0].allocate_resource(min(10, self.slices[0].allocated_resources + 1))
            self.slices[1].allocate_resource(max(0, self.slices[1].allocated_resources - 1))

        elif action == 2:
            self.action2 += 1

            # move one resource to slice 1
            self.slices[0].allocate_resource(max(0, self.slices[0].allocated_resources - 1))
            self.slices[1].allocate_resource(min(10, self.slices[1].allocated_resources + 1))
        else:
            # action was not recognized
            raise ValueError("Action value was not recognized")

        # move the system
        for i in range(2):
            self.slices[i].step()
            self.slices[i].reset_temp_stats()
            tmp_state.append(len(self.slices[i]))
        SimGlobals.NET_TIMESLOT_STEP += 1

        self.state = tmp_state
        self.state.append(self.slices[0].allocated_resources)
        self.state.append(self.slices[1].allocated_resources)

        info = {
            "episode": 0,
            "stats": stats,
        }

        done = self.end
        if self.end:
            self.end = False

        return np.array(self.state), reward_func(self.state[2], self.state[3]), done, info

    def reset(self, **kwargs):
        self.action0 = 0
        self.action1 = 0
        self.action2 = 0
        lambda_1, _ = self.generators[0].reset()
        lambda_2, _ = self.generators[1].reset()
        self.slices = []
        for i in range(2):
            self.slices.append(NetQueue())
        self.state = [lambda_1, lambda_2, len(self.slices[0]), len(self.slices[1]), self.slices[0].allocated_resources, self.slices[1].allocated_resources ]
        return np.array(self.state)

    def set_end(self):
        self.end = True
