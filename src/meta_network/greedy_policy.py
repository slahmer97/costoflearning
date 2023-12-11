import numpy as np


class OnlyLearningPlanePolicy:
    @staticmethod
    def choose_action(slice0_qsize, slice1_qsize, slice1_up, slice1sp, learning_pp, avg0, avg1):
        return 0, 0, 15


class GreedyPolicy:

    @staticmethod
    def choose_action(slice0_qsize, slice1_qsize, slice1_up, slice1sp, learning_pp, avg0, avg1):
        remaining = 15
        r2 = min(learning_pp * 3, 6)
        remaining -= r2

        coeif0, coeif1 = avg0 / 1000.0, avg1 / 60.0

        sum = coeif0 + coeif1

        coeif0, coeif1 = coeif0 / sum, coeif1 / sum

        r0 = int(coeif0 * remaining)
        remaining -= r0

        r1 = remaining
        #print(r0,r1,r2)
        return r0, r1, r2



