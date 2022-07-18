from network.flow_gen import OnOffFlowGenerator, UniformFlowGenerator
import numpy as np

from network.globsim import SimGlobals
from network.netqueue import NetQueue


class Network:

    def __init__(self):
        self.uniformconf = [
            {
                "packet_size": 512,
                "req_delay": 10.0,
                "max_delay": np.infty,
                "rate": 64000,
                "flow_class": 'audio',
                "flow_model": 'unicast',
                "flow_performance": 'strict',
                "slice": 0
            },
            {
                "packet_size": 512,
                "req_delay": 30.0,
                "max_delay": np.infty,
                "rate": 384000,
                "flow_class": 'video',
                "flow_model": 'unicast',
                "flow_performance": 'linear',
                "slice": 0
            },
            {
                "packet_size": 512,
                "req_delay": 30.0,
                "max_delay": np.infty,
                "rate": 384000,
                "flow_class": 'video',
                "flow_model": 'unicast',
                "flow_performance": 'linear',
                "slice": 0
            }

        ]

        self.onoffconfs = [
            {
                "packet_size": 512,
                "req_delay": 7.5,
                "max_delay": 7.5,
                "rate": 64000,
                "flow_class": 'critical-audio',
                "flow_model": 'unicast',
                "flow_performance": 'strict',
                "slice": 1
            },
            {
                "packet_size": 512,
                "req_delay": 100,
                "max_delay": 100,
                "rate": 384000,
                "flow_class": 'critical-video',
                "flow_model": 'unicast',
                "flow_performance": 'strict',
                "slice": 1
            }

        ]


        self.slices = []
        for i in range(2):
            self.slices.append(NetQueue())

        self.generators = []

        for conf in self.onoffconfs:
            self.generators.append(OnOffFlowGenerator(**conf))

        for conf in self.uniformconf:
            self.generators.append(UniformFlowGenerator(**conf))

    def step(self, action=None):
        for generator in self.generators:
            a = generator.step()
            self.slices[generator.slice].enqueue(a)
            print("flow={} -- packets-count={}".format(generator.flow_id, len(a)))

        for i in range(2):
            self.slices[i].step()
            self.slices[i].reset_temp_stats()
        SimGlobals.NET_TIMESLOT_STEP += 1

    def reset(self):
        self.slices[0].allocated_resources = 5
        self.slices[1].allocated_resources = 95

    def get_network_state(self):
        pass