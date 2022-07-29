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
        #max_timeslots = self.max_delay / SimGlobals.NET_TIMESLOT_DURATION
        #print("{} <= {}".format(SimGlobals.NET_TIMESLOT_STEP - self.generated_at, max_timeslots))
        #assert SimGlobals.NET_TIMESLOT_STEP - self.generated_at <= max_timeslots
        return False
        return SimGlobals.NET_TIMESLOT_STEP - self.generated_at >= max_timeslots
