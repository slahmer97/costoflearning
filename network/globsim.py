class SimGlobals:
    NET_TIMESLOT_STEP = 0
    NET_TIMESLOT_DURATION_S = 0.0005  # in s
    RESOURCES_COUNT = 10
    BANDWIDTH_PER_RESOURCE = 1000000# byte per second // 0.1 mbps
    TOTAL_LINK_BANDWIDTH = RESOURCES_COUNT * BANDWIDTH_PER_RESOURCE

    EXPERIENCE_SIZE = 1500  # in bits
    flow_counter = 0

    audio_bitrate = [4000, 25000]  # [b/s]
    video_bitrate = [32000, 384000]  # [b/s]

    # Probabilities TODO tobe changed
    P00 = 0.7
    P01 = 0.3
    P11 = 0.7
    P10 = 0.3

    @staticmethod
    def reset():
        SimGlobals.NET_TIMESLOT_STEP = 0
        SimGlobals.NET_TIMESLOT_DURATION_S = 0.0005  # in s
        SimGlobals.RESOURCES_COUNT = 10
        SimGlobals.BANDWIDTH_PER_RESOURCE = 1000000 # byte per second // 0.1 mbps
        SimGlobals.TOTAL_LINK_BANDWIDTH = SimGlobals.RESOURCES_COUNT * SimGlobals.BANDWIDTH_PER_RESOURCE

        SimGlobals.EXPERIENCE_SIZE = 1500  # in bits
        SimGlobals.flow_counter = 0

        SimGlobals.audio_bitrate = [4000, 25000]  # [b/s]
        SimGlobals.video_bitrate = [32000, 384000]  # [b/s]

        # Probabilities TODO tobe changed
        SimGlobals.P00 = 0.7
        SimGlobals.P01 = 0.3
        SimGlobals.P11 = 0.7
        SimGlobals.P10 = 0.3
