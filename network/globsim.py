class SimGlobals:
    NET_TIMESLOT_STEP = 0
    NET_TIMESLOT_DURATION_S = 0.001  # in s
    RESOURCES_COUNT = 10
    BANDWIDTH_PER_RESOURCE = 1000000  # 0.1 mBps
    dropped_before_enqueue_packets = []
    dropped_after_enqueue_packets = []
    served_packets = []

    flow_counter = 0

    audio_bitrate = [4000, 25000]  # [b/s]
    video_bitrate = [32000, 384000]  # [b/s]

    # Probabilities TODO tobe changed
    P00 = 0.8
    P01 = 0.2
    P11 = 0.7
    P10 = 0.3
