import numpy as np

from .globsim import SimGlobals


class Packet:
    def __init__(self, flow_id, required_delay, max_delay, size=1500):
        self.flow_id = flow_id

        self.generated_at = None
        self.served_at = None

        # constraints
        self.max_delay = max_delay  # in ms
        self.required_delay = required_delay  # in ms

        assert self.max_delay >= self.required_delay

        self.size = size  # in bit

    def get_packet_state(self):

        req_delay_max_slot = int(self.required_delay / SimGlobals.NET_TIMESLOT_DURATION_S)

        if self.max_delay == np.infty:
            if SimGlobals.NET_TIMESLOT_STEP < self.generated_at + req_delay_max_slot:
                return "soft", np.infty
            else:
                return "hard", np.infty
        else:
            max_delay_max_slot = int(self.max_delay / SimGlobals.NET_TIMESLOT_DURATION_S)

        assert req_delay_max_slot >= 1
        assert max_delay_max_slot >= 1

        assert SimGlobals.NET_TIMESLOT_STEP <= self.generated_at + req_delay_max_slot
        assert SimGlobals.NET_TIMESLOT_STEP <= self.generated_at + max_delay_max_slot

        # deadline already passed
        if SimGlobals.NET_TIMESLOT_STEP >= self.generated_at + max_delay_max_slot:
            return "dead", - abs(SimGlobals.NET_TIMESLOT_STEP - self.generated_at - max_delay_max_slot)

        # soft deadline 1 duration is still less than the requested one
        if SimGlobals.NET_TIMESLOT_STEP < self.generated_at + req_delay_max_slot:
            return "soft", abs(SimGlobals.NET_TIMESLOT_STEP - self.generated_at - max_delay_max_slot)

        # time is between req and max
        if self.max_delay != self.required_delay:
            return "hard", abs(SimGlobals.NET_TIMESLOT_STEP - self.generated_at - req_delay_max_slot)
        else:
            i = SimGlobals.NET_TIMESLOT_STEP
            print("unknozn")
    def is_dead(self):
        if self.max_delay == np.Inf:
            return False

        max_timeslot = int(self.max_delay / SimGlobals.NET_TIMESLOT_DURATION_S)

        assert max_timeslot >= 1

        assert SimGlobals.NET_TIMESLOT_STEP <= self.generated_at + max_timeslot

        do_not_drop = SimGlobals.NET_TIMESLOT_STEP < self.generated_at + max_timeslot

        # if not do_not_drop:
        #    print("Packet has been dropped! : inserted: {} -- step: {} -- max_timeslot: {}".format(self.generated_at, SimGlobals.NET_TIMESLOT_STEP, max_timeslot))

        return not do_not_drop
