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


class PolicySelector:
    def __init__(self, g_eps=0, g_eps_decay=0, g_eps_min=0, use=False):
        self.use = use
        self.g_eps = g_eps
        self.g_eps_decay = g_eps_decay
        self.g_eps_min = g_eps_min

        self.step = 0

        self.greedy = 0
        self.non_greedy = 0
        self.slope = - 0.0000018

    def reset_gepsilon(self, start=0.2, end=60):
        self.slope = np.log(self.g_eps_min / start) / end
        self.i = 0
        self.step = 0
        self.start = start
        self.g_eps = start

        # self.slope = (self.g_eps_min - 0.20) / float(end * 1000)

    def choose_action(self):
        if not self.use:
            return 1

        self.step += 1

        #if self.step % 100 == 0:
        #    return 2
        #return 1

        if self.step % 1000 == 0:
            print(f"step:{self.step}")
            # self.g_eps = max(self.g_eps_min, self.g_eps * self.g_eps_decay)
            # self.g_eps = -7.96e-7 * self.step + 3.0 / 15.0
            # self.g_eps = max(self.g_eps, self.g_eps_min)
            currentStep = int(self.step / 1000)
            self.g_eps = self.start * np.exp(self.slope * currentStep)

            # self.g_eps = self.slope * self.step + 0.18
            self.g_eps = max(self.g_eps, self.g_eps_min)

        if np.random.random(1)[0] > self.g_eps or self.step <= 1:
            self.non_greedy += 1
            return 1
        self.greedy += 1

        return 2
