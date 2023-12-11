import numpy as np
import time


def context_to_id(max1, max2, resources):
    counter = 0
    hash = {}
    for i in range(1, max1):
        for j in range(1, max2):
            hash[(i, j, resources)] = counter
            counter += 1

    return hash


def sample_tasks():
    def generate_tasks():
        P_0_00 = np.random.uniform(0.05, 0.95, 1)[0]
        P_0_01 = 1 - P_0_00

        P_0_11 = np.random.uniform(0.05, 0.95, 1)[0]
        P_0_10 = 1 - P_0_11

        P_1_00 = np.random.uniform(0.05, 0.95, 1)[0]
        P_1_01 = 1 - P_1_00

        P_1_11 = np.random.uniform(0.05, 0.95, 1)[0]
        P_1_10 = 1 - P_1_11

        on_0 = P_0_01 / (P_0_01 + P_0_10)

        on_1 = P_1_01 / (P_1_01 + P_1_10)

        max_1 = np.random.randint(2, int(14 / on_0), 1)[0]

        max_2 = int(max(int(15 - max_1 * on_0), 1) / on_1)
        load = max_1 * on_0 + max_2 * on_1
        if 15.1 < load or load < 13.9:
            return None
        print(load)
        print()
        print()
        cfg = {
            "max_users:0": max_1,
            "max_users:1": max_2,
            "transitions": [
                [
                    [P_0_00, P_0_01],
                    [P_0_10, P_0_11]
                ]
                ,

                [
                    [P_1_00, P_1_01],
                    [P_1_10, P_1_11]
                ]

            ],
            "resources_count": 15,
            "cost_weights": [0.25, 0.75],
            "load": load
        }
        return cfg

    confs = []
    count = 0
    while count <= 128:
        try:
            tmp = generate_tasks()
            if tmp is not None:
                confs.append(tmp)
                count += 1
        except:
            print("---exception")
        time.sleep(0.1)
    for s in confs:
        print(s,",")
    return confs
