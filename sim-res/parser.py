import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set()

file = pd.read_csv("reward", delimiter=" ")

greedyActions = file['greedy'].to_numpy()
reward1 = file['reward1'].str.replace('(', '').str.replace(')', '').to_numpy(dtype=float)

reward_distribution_data = pd.read_csv("reward_distribution.csv")

data = []
for (count, val) in zip(greedyActions, reward1):
    for _ in range(count):
        point = -1.0 * val / count
        if point < 0:
            print(point)
        data.append(point)
data = - reward_distribution_data["reward"].to_numpy()

sns.kdeplot(data=data, cumulative=True)
plt.ylabel("Density")
plt.xlim(0.0, 1.0)
plt.xlabel("Overhead per Greedy Action ( -Reward )")
import tikzplotlib

tikzplotlib.save("../figures/overheadPerGreedyAction.tex")
print(tikzplotlib.Flavors.latex.preamble())
plt.savefig("../figures/overheadPerGreedyAction.png")
plt.show()
# nonGreedyActions = file['nonGreedy'].to_numpy()

# reward2 = file['reward2'].str.replace('(', '').str.replace(')', '').to_numpy(dtype=float)

# average_reward_per_actionGreedy = reward1 / greedyActions
# average_reward_per_NonactionGreedy = reward2 / nonGreedyActions

# plt.plot(np.arange(0,2000)*1000, average_reward_per_actionGreedy, color="blue", label="average reward using GreedyP")
# plt.plot(np.arange(0,2000)*1000, average_reward_per_NonactionGreedy, color="red", label="average reward using RL")
# plt.legend()

# plt.savefig("average-reward-per-action.png")
# plt.show()
