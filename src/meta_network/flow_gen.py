from src.meta_network.ActiveUserModel import ActiveUsers
from src.meta_network.packet import Packet


class FlowGenerator:
    def __init__(self, sim, packet_size=512, req_delay=0.075, max_delay=0.075, rate=25000,
                 flow_class='critical', flow_model='unicast', flow_performance='strict', slice=0):
        self.slice = slice
        self._sim = sim
        self.rate = int(rate * self._sim.NET_TIMESLOT_DURATION_S / packet_size)
        self.packet_size = packet_size
        self.req_delay = req_delay
        self.max_delay = max_delay
        self.flow_class = flow_class
        self.flow_model = flow_model
        self.flow_performance = flow_performance
        self._sim.flow_counter += 1
        self.flow_id = self._sim.flow_counter

    def step(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class AggregateOnOffFlowGenerator(FlowGenerator):
    def __init__(self, sim, max_users=10, packet_size=512, req_delay=0.075, max_delay=0.075, on_rate=64000
                 , flow_class='critical', flow_model='unicast', flow_performance='strict', slice=0):
        super().__init__(sim, packet_size, req_delay, max_delay, on_rate, flow_class, flow_model, flow_performance,
                         slice)

        self.max_users = max_users
        self.active_users_model = ActiveUsers(sim, max_users=self.max_users, process_name='slice-{}'.format(slice),
                                              slice=slice)

        # s_{t} = s
        self.current_state = None
        expected_val, variance = self.active_users_model.get_mean_var()
        # return transition probabilities of the next state given the current state
        # Pr(s_{t+1} | s_{t} = s)
        self.distribution_over_next_state = None
        print("[+] New AggregateOnOffFlowGenerator has been created [{}]\n"
              "\t max users: {}\n"
              "\t slice : {}\n"
              "\t rate : {} per sec\n"
              "\t rate : {} packet per slot\n"
              "\t packet size: {}\n"
              "\t required delay: {}\n"
              "\t max delay: {}\n"
              "\t flow class: {}\n"
              "\t flow model: {}\n"
              "\t flow perf: {}\n"
              "\t Exp[users]: {} -- Var[users]: {}\n"
              .format(self.flow_id, self.max_users, self.slice, on_rate, self.rate, self.packet_size,
                      self.req_delay, self.max_delay,
                      self.flow_class, self.flow_model, self.flow_performance, expected_val, variance))

    def step(self):
        ret = []
        for _ in range(self.current_state * self.rate):
            p = Packet(self._sim, self.flow_id, self.req_delay, self.max_delay, size=self.packet_size)
            ret.append(p)

        self.current_state, self.distribution_over_next_state = self.active_users_model.step()
        return ret, self.current_state, self.distribution_over_next_state

        # ret = []
        # if self.current_state == 'on':
        #    sampled_rate = max(1, self.rate)
        #    for i in range(sampled_rate):
        #        p = Packet(self.flow_id, self.req_delay, self.max_delay, size=self.packet_size)
        #        ret.append(p)

        # self.random_walk()
        # return ret

    def reset(self):
        self.current_state, self.distribution_over_next_state = self.active_users_model.reset()
        return self.current_state, self.distribution_over_next_state

    def display_steady_state(self):
        pass

    def random_walk(self):
        pass



