import numpy as np
from pydtmc import MarkovChain, plot_graph
from .globsim import SimGlobals

from matplotlib import pyplot as plt


class ActiveUsers:
    def __init__(self, max_users=2, process_name='A'):
        self.process_name = process_name
        self.max_users = max_users
        self.transitions = np.zeros(shape=(max_users + 1, max_users + 1))
        self.current_state = None
        self.states = np.array([i for i in range(self.max_users + 1)])

        for i in range(0, max_users + 1):
            sum = 0
            for j in range(0, max_users + 1):
                self.transitions[i][j] = self.get_prob(j, i)
                sum += self.transitions[i][j]
            #print(sum)
        # print(self.transitions)
        self.mc = MarkovChain(self.transitions,
                              ['{}{}'.format(self.process_name, i) for i in range(0, self.max_users + 1)])

    def get_prob(self, j, h):

        l_min = max(0, j - self.max_users + h)
        l_max = min(h, j)
        sum = 0.0
        N = self.max_users
        for l in range(l_min, l_max + 1):
            term1 = np.math.comb(h, l) * np.math.comb(N - h, j - l)
            term2 = np.power(SimGlobals.P11, l) * np.power(SimGlobals.P01, j - l) * np.power(SimGlobals.P10, h - l) * \
                    np.power(SimGlobals.P00, self.max_users - h - j + l)
            sum += term1 * term2
        return sum

    def reset(self):
        self.current_state = np.random.choice(self.states, p=self.mc.steady_states[0])
        return self.current_state, self.transitions[self.current_state]

    def step(self):
        self.current_state = np.random.choice(self.states, p=self.transitions[self.current_state])
        return self.current_state, self.transitions[self.current_state]

    def get_mean(self):
        steady_state = self.mc.steady_states[0]
        states = np.array([i for i in range(self.max_users + 1)])
        ret = np.dot(steady_state, states)
        return ret

    def simulate(self):
        current_state = np.random.choice(self.states, p=self.mc.steady_states[0])
        history = [current_state]
        index = [0]
        for i in range(1, 1000):
            current_state = np.random.choice(self.states, p=self.transitions[current_state])
            history.append(current_state)
            index.append(i)

        plt.plot(index, history)
        plt.show()

    def display(self):
        plot_graph(self.mc)
        plt.show()
