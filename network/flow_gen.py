from pydtmc import MarkovChain
import numpy as np

from network.globsim import SimGlobals
from network.packet import Packet


class FlowGenerator:
    def __init__(self, packet_size=512, req_delay=0.075, max_delay=0.075, rate=25000,
                 flow_class='critical', flow_model='unicast', flow_performance='strict', slice=0):
        self.slice = slice
        self.rate = int(rate * SimGlobals.NET_TIMESLOT_DURATION_S / packet_size)
        self.packet_size = packet_size
        self.req_delay = req_delay
        self.max_delay = max_delay
        self.flow_class = flow_class
        self.flow_model = flow_model
        self.flow_performance = flow_performance
        SimGlobals.flow_counter += 1
        self.flow_id = SimGlobals.flow_counter

        print("[+] New flow has been created [{}]\n"
              "\t packet size: {}\n"
              "\t required delay: {}\n"
              "\t max delay: {}\n"
              "\t flow class: {}\n"
              "\t flow model: {}\n"
              "\t flow perf: {}".format(self.flow_id, self.packet_size, self.req_delay, self.max_delay,
                                        self.flow_class, self.flow_model, self.flow_performance))

    def step(self):
        raise NotImplementedError

    def reset(self, init_state):
        raise NotImplementedError


class OnOffFlowGenerator(FlowGenerator):
    def __init__(self, transitions=None, packet_size=512, req_delay=0.075, max_delay=0.075, rate=64000.0
                 , flow_class='critical', flow_model='unicast', flow_performance='strict', slice=0):
        super().__init__(packet_size, req_delay, max_delay, rate, flow_class, flow_model, flow_performance, slice)
        if transitions is None:
            transitions = [[0.9, 0.1], [0.1, 0.9]]
        self.transitions = transitions

        self.mc = MarkovChain(self.transitions, ['on', 'off'])
        self.current_state = np.random.choice(['on', 'off'], p=self.mc.steady_states[0])
        print("\t rate: {}".format(self.rate))

    def step(self):
        ret = []
        if self.current_state == 'on':
            sampled_rate = max(1, self.rate)
            for i in range(sampled_rate):
                p = Packet(self.flow_id, self.req_delay, self.max_delay, size=self.packet_size)
                ret.append(p)

        self.random_walk()
        return ret

    def reset(self, init_state=None):
        self.current_state = np.random.choice(['on', 'off'], p=self.mc.steady_states[0])

    def display_steady_state(self):
        print("steady_state: {}".format(self.mc.steady_states[0]))

    def random_walk(self):
        assert self.current_state in ['on', 'off'], "random walk"
        tmp = None
        if self.current_state == 'on':
            tmp = np.random.choice(['on', 'off'], p=self.transitions[0])
        elif self.current_state == 'off':
            tmp = np.random.choice(['on', 'off'], p=self.transitions[1])
        assert tmp is not None

        self.current_state = tmp


class UniformFlowGenerator(FlowGenerator):
    def __init__(self, packet_size=512, req_delay=0.075, max_delay=0.075, rate=64000.0
                 , flow_class='critical', flow_model='unicast', flow_performance='strict', slice=0):
        super().__init__(packet_size, req_delay, max_delay, rate, flow_class, flow_model, flow_performance, slice)

    def step(self):
        ret = []
        sampled_rate = max(1, self.rate)
        for i in range(sampled_rate):
            p = Packet(self.flow_id, self.req_delay, self.max_delay, size=self.packet_size)
            ret.append(p)

        return ret

    def reset(self, init_state=None):
        pass
