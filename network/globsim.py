import numpy as np


class SimGlobals:
    NET_TIMESLOT_STEP = None
    NET_TIMESLOT_DURATION_S = None  # in s
    RESOURCES_COUNT = None
    BANDWIDTH_PER_RESOURCE = None  # byte per second // 0.1 mbps
    TOTAL_LINK_BANDWIDTH = None

    EXPERIENCE_SIZE = None  # in bits
    flow_counter = None

    Transitions = [
        np.array(
            [
                [0.5, 0.5],
                [0.92, 0.08]
            ]
        ),
        np.array(
            [
                [0.5, 0.5],
                [0.5, 0.5]
            ]
        )
    ]

    success_prob = 1.0
    # capacity_seq = [0.9878908956925939, 0.9791408975149585, 0.8300037254456502, 0.8973736207129992, 0.9185000230967884,
    #                0.9690193446267908, 0.8053623484906383, 0.9127649177036951, 0.963916082701487, 0.8462046386053668,
    #                0.9399992720743086, 0.954639025235394, 0.8257302290416382, 0.9748797286963092, 0.9362267826412424,
    #                0.8054427332030936, 0.892833963222041, 0.8846928748459606, 0.9351130630343018, 0.8437296158757011,
    #                0.8708249098793576, 0.8614997801689942, 0.934573414889476, 0.83609028843595, 0.9698478264162679,
    #                0.9512961607953276, 0.8583359627641207, 0.9362721411853542, 0.8880833619883496, 0.9079393268431515]
    # capacity_seq = [0.9964156373752202, 0.8909746144689621, 0.9306387483820725, 0.8584782290559012, 0.8810284247376136,
    #                0.9673536886706504, 0.8936067777551695, 0.9910602614652789, 0.9155053850635161, 0.8868139437281872,
    #                0.9050191839829524, 0.8572830133853764, 0.9481311396771338, 0.9248044163050405, 0.875990822237412,
    #                0.8978794294309269, 0.9555140462758999, 0.9472757635945788, 0.8533391226404007, 0.9816395850957593,
    #                0.9303412786776923, 0.8783202184198073, 0.8729118394170086, 0.9111763458380514, 0.8479689229980798,
    #                0.9293793033337236, 0.8836767682705863, 0.8757585068432472, 0.8644556663852767, 0.8494230902400826]

    # capacity_seq = [0.9999, 0.88, 0.9, 0.89, 0.95, 0.91, 0.88, 0.99, 0.94, 0.91, 0.899, 1.0]

    capacity_seq = [0.95, 0.89, 0.9, 0.885, 0.89, 0.882, 0.875, 0.86, 0.865, 0.86, 0.899, 0.91]

    my_counter = 0

    @staticmethod
    def init_success_prob(prob):
        SimGlobals.success_prob = prob

    @staticmethod
    def update_success_prob(dqn, learning_queue, greedySelector):
        if SimGlobals.NET_TIMESLOT_STEP % 500000 == 0:
            # SimGlobals.success_prob = SimGlobals.capacity_seq[SimGlobals.my_counter]
            SimGlobals.my_counter += 1
            # learning_queue.stop = True

            dqn.reset_epsilon(1.0, 100)
            #greedySelector.reset_gepsilon(100)
            dqn.reset_mem()
            learning_queue.reset()
            print("Updated Greedy Selection")
            # print("Changed Success Probability: {}".format(SimGlobals.success_prob))

    @staticmethod
    def send_success():
        return True  # np.random.random(1)[0] <= SimGlobals.success_prob

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
        if pkt_latency < 50:
            return 0.0
        else:
            return - (1.0 / 70.0) * pkt_latency
