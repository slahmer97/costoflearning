import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def plotAverageReward(files=[0, 1, 2, 3]):
    for i in files:
        data = pd.read_csv("{}.log".format(i), delimiter=" ")
        reward = (data['reward1'].str.replace('(', '').str.replace(')', '').astype('float64') +
                  data['reward2'].str.replace('(', '').str.replace(')', '').astype('float64')
                  ).expanding().mean().to_numpy(dtype=float)


        plt.plot(reward)


    plt.savefig("../figures/averageReward.png")
    plt.show()


plotAverageReward()
