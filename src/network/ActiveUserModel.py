import numpy as np
from pydtmc import MarkovChain, plot_graph
from .globsim import SimGlobals

from matplotlib import pyplot as plt


class ActiveUsers:
    def __init__(self, max_users=2, process_name='A', slice=0):
        self.process_name = process_name
        self.max_users = max_users
        self.transitions = np.zeros(shape=(max_users + 1, max_users + 1))
        self.current_state = None
        self.states = np.array([i for i in range(self.max_users + 1)])
        self.slice = slice
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
            term2 = np.power(SimGlobals.Transitions[self.slice][1][1], l) * np.power(SimGlobals.Transitions[self.slice][0][1], j - l) * np.power(SimGlobals.Transitions[self.slice][1][0], h - l) * \
                    np.power(SimGlobals.Transitions[self.slice][0][0], self.max_users - h - j + l)
            sum += term1 * term2
        return sum

    def reset(self):
        self.current_state = np.random.choice(self.states, p=self.mc.steady_states[0])
        return self.current_state, self.transitions[self.current_state]

    def step(self):
        self.current_state = np.random.choice(self.states, p=self.transitions[self.current_state])
        return self.current_state, self.transitions[self.current_state]

    def get_mean_var(self):
        steady_state = self.mc.steady_states[0]
        i = np.array([i for i in range(self.max_users + 1)])
        i_2 = np.array([pow(i, 2) for i in range(self.max_users + 1)])

        expected_val = np.dot(steady_state, i)
        variance = np.dot(steady_state, i_2) - pow(expected_val, 2)

        return expected_val, variance

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
