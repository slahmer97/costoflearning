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

        self.size = size  # in bit

    def is_dead(self):
        if self.max_delay == np.Inf:
            return False

        max_timeslot = int(self.max_delay / SimGlobals.NET_TIMESLOT_DURATION_S)

        assert max_timeslot >= 1

        assert SimGlobals.NET_TIMESLOT_STEP <= self.generated_at + max_timeslot

        do_not_drop = SimGlobals.NET_TIMESLOT_STEP < self.generated_at + max_timeslot

        #if not do_not_drop:
        #    print("Packet has been dropped! : inserted: {} -- step: {} -- max_timeslot: {}".format(self.generated_at, SimGlobals.NET_TIMESLOT_STEP, max_timeslot))

        return not do_not_drop

