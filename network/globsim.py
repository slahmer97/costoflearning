class SimGlobals:
    NET_TIMESLOT_STEP = 0
    NET_TIMESLOT_DURATION = 10  # in ms
    NET_TIMESLOT_DURATION_S = 10 * 0.001  # in ms
    RESOURCES_COUNT = 100
    BANDWIDTH_PER_RESOURCE = 1000  # 0.1 mbps
    dropped_before_enqueue_packets = []
    dropped_after_enqueue_packets = []
    served_packets = []

    flow_counter = 0

    audio_bitrate = [4000, 25000]  # [b/s]
    video_bitrate = [32000, 384000]  # [b/s]

    # Probabilities TODO tobe changed
    P00 = 0.1
    P01 = 0.9
    P11 = 0.5
    P10 = 0.5
