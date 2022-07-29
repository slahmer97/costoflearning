from network.flow_gen import OnOffFlowGenerator, UniformFlowGenerator, AggregateOnOffFlowGenerator
import numpy as np

from network.globsim import SimGlobals
from network.netqueue import NetQueue


class Network:

    def __init__(self):
        self.state = None
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

        self.slices = []
        for i in range(2):
            self.slices.append(NetQueue())

        res = 5
        self.slices[0].allocate_resource(res)
        self.slices[1].allocate_resource(10-res)

        self.generators = []

        for conf in self.slice_0_flows:
            self.generators.append(AggregateOnOffFlowGenerator(**conf))

        for conf in self.slice_1_flows:
            self.generators.append(AggregateOnOffFlowGenerator(**conf))

    def step(self, action=None):
        tmp_state = []
        stats = []
        for generator in self.generators:
            # generate the packets before running the activeuser model to the next state, then makes the move
            a, active_user, _ = generator.step()
            self.slices[generator.slice].enqueue(a)
            tmp_state.append(active_user)
            stats.append(len(a) * 512)
        # Apply the new resource allocation according to the action=action

        # move the system
        for i in range(2):
            self.slices[i].step()
            self.slices[i].reset_temp_stats()
            tmp_state.append(len(self.slices[i]))
        SimGlobals.NET_TIMESLOT_STEP += 1

        self.state = tmp_state
        return np.array(self.state), 0.0, stats

    def reset(self):
        lambda_1, _ = self.generators[0].reset()
        lambda_2, _ = self.generators[1].reset()
        self.state = [lambda_1, lambda_2, len(self.slices[0]), len(self.slices[1])]
        return np.array(self.state)

