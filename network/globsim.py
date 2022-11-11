class SimGlobals:
    NET_TIMESLOT_STEP = None
    NET_TIMESLOT_DURATION_S = None  # in s
    RESOURCES_COUNT = None
    BANDWIDTH_PER_RESOURCE = None  # byte per second // 0.1 mbps
    TOTAL_LINK_BANDWIDTH = None

    EXPERIENCE_SIZE = None  # in bits
    flow_counter = None

    P00 = None
    P01 = None
    P11 = None
    P10 = None

    @staticmethod
    def reset():
        SimGlobals.NET_TIMESLOT_STEP = 0
        SimGlobals.NET_TIMESLOT_DURATION_S = 0.001  # in s
        SimGlobals.RESOURCES_COUNT = 15
        SimGlobals.BANDWIDTH_PER_RESOURCE = 513000  # byte per second // 0.1 mbps
        SimGlobals.TOTAL_LINK_BANDWIDTH = SimGlobals.RESOURCES_COUNT * SimGlobals.BANDWIDTH_PER_RESOURCE

        SimGlobals.EXPERIENCE_SIZE = 1500  # in bits
        SimGlobals.flow_counter = 0

        SimGlobals.audio_bitrate = [4000, 25000]  # [b/s]
        SimGlobals.video_bitrate = [32000, 384000]  # [b/s]

        # Probabilities TODO tobe changed
        SimGlobals.P00 = 0.5
        SimGlobals.P01 = 0.5
        SimGlobals.P11 = 0.5
        SimGlobals.P10 = 0.5
