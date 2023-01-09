import numpy as np


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

    success_prob = 1.0
    capacity_seq = [0.9878908956925939, 0.9791408975149585, 0.8300037254456502, 0.8973736207129992, 0.9185000230967884,
                    0.9690193446267908, 0.8053623484906383, 0.9127649177036951, 0.963916082701487, 0.8462046386053668,
                    0.9399992720743086, 0.954639025235394, 0.8257302290416382, 0.9748797286963092, 0.9362267826412424,
                    0.8054427332030936, 0.892833963222041, 0.8846928748459606, 0.9351130630343018, 0.8437296158757011,
                    0.8708249098793576, 0.8614997801689942, 0.934573414889476, 0.83609028843595, 0.9698478264162679,
                    0.9512961607953276, 0.8583359627641207, 0.9362721411853542, 0.8880833619883496, 0.9079393268431515]

    my_counter = 0

    @staticmethod
    def init_success_prob(prob):
        SimGlobals.success_prob = prob

    @staticmethod
    def update_success_prob(dqn, learning_queue, greedySelector):
        if SimGlobals.NET_TIMESLOT_STEP % 400000 == 0:
            SimGlobals.success_prob = SimGlobals.capacity_seq[SimGlobals.my_counter]
            SimGlobals.my_counter += 1

            if SimGlobals.NET_TIMESLOT_STEP != 0:
                dqn.reset_mem()
                learning_queue.reset()
                dqn.reset_epsilon(0.3)
                greedySelector.reset_gepsilon(0.08)
            print("Changed Success Probability: {}".format(SimGlobals.success_prob))

    @staticmethod
    def send_success():
        return np.random.random(1)[0] <= SimGlobals.success_prob

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
        SimGlobals.success_prob = 1.0

        # Probabilities TODO tobe changed
        SimGlobals.P00 = 0.5
        SimGlobals.P01 = 0.5

        SimGlobals.P11 = 0.2
        SimGlobals.P10 = 0.8

    @staticmethod
    def urllc_cost(pkt_latency):
        if pkt_latency < 60:
            return 0
        elif pkt_latency >= 60:
            return - (1.0 / 100.0) * pkt_latency

