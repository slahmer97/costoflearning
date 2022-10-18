from collections import deque
import matplotlib.pyplot as plt
import numpy as np


class StatCollector:

    def __init__(self):
        max_plot = 1000
        self.average_on = 20
        self.incoming_packets1 = deque(maxlen=max_plot)
        self.incoming_packets2 = deque(maxlen=max_plot)

        self.resources1 = deque(maxlen=max_plot)
        self.resources2 = deque(maxlen=max_plot)

        self.generated_packets1 = deque(maxlen=max_plot)
        self.generated_packets2 = deque(maxlen=max_plot)

        self.epsilons = deque(maxlen=max_plot)

        self.queue1 = deque(maxlen=max_plot)
        self.queue2 = deque(maxlen=max_plot)
        self.learning_queue = deque(maxlen=max_plot)

        self.reward_list = deque(maxlen=self.average_on)
        self.average_reward = []
        plt.ion()
        self.fig, self.axs = plt.subplots(5)

    def push_episodic_stats(self, cum_reward, epsilon, average_q_learner):
        self.reward_list.append(cum_reward)
        self.average_reward.append(np.mean(self.reward_list))
        self.epsilons.append(epsilon)

        self.learning_queue.append(average_q_learner)


    """
    @q: size of the queue
    @r: number of resources
    @gp: number of generated packet with size of 512.
    """

    def push_data(self, q1, q2, r1, r2, gp1=0, gp2=0):
        self.queue1.append(q1)
        self.queue2.append(q2)

        self.resources1.append(r1)
        self.resources2.append(r2)

        self.generated_packets1.append(gp1)
        self.generated_packets2.append(gp2)

    def plot(self):
        """
        self.axs[0].cla()
        self.axs[0].plot(self.average_reward)
        self.axs[0].set_title('Average Sum of Reward over 1000 step')

        self.axs[1].cla()
        self.axs[1].plot(self.epsilons)
        self.axs[1].set_title('epsilon')

        self.axs[2].cla()
        self.axs[2].plot(self.queue1, label='q1')
        self.axs[2].plot(self.queue2, label='q2')
        # self.axs[2].plot(self.learning_queue, label='q-ler')
        self.axs[2].legend()
        self.axs[2].set_title('Queue Size')

        self.axs[3].cla()
        self.axs[3].plot(self.resources1, label='res1')
        self.axs[3].plot(self.resources2, label='res2')
        self.axs[3].legend()
        self.axs[3].set_title('Resource Allocation')

        self.axs[4].cla()
        self.axs[4].plot(self.learning_queue)
        self.axs[4].set_title('queue-learner')



        self.axs[4].cla()
        self.axs[4].plot(self.generated_packets1, label='gp1')
        self.axs[4].plot(self.generated_packets2, label='gp2')
        self.axs[4].legend()
        self.axs[4].set_title('Generated Packets')
        plt.tight_layout()
        plt.pause(0.001)
        """

        print('Reward: {} -- EPS: {}'.format(self.average_reward[-1], self.epsilons[-1]))
        print("Average: R1={} -- R2={}".format(np.mean(self.resources1), np.mean(self.resources2)))


