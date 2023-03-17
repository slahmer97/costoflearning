import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Set the style of the plot
from meta_network.tasks import Tasks

sns.set_style("whitegrid")
sns.set_context('paper')

task_sampler = Tasks()
init_lambda0 = []
init_lambda1 = []
lamda0 = []
lamda1 = []
for t in task_sampler.tasks:
    l0 = float(t["transitions"][0][0][1] / (t["transitions"][0][0][1] + t["transitions"][0][1][0])) * t["max_users:0"]
    l1 = float(t["transitions"][1][0][1] / (t["transitions"][1][0][1] + t["transitions"][1][1][0])) * t["max_users:1"]

    #print(l0+l1, t["load"])
    init_lambda0.append(l0)
    init_lambda1.append(l1)
# Create a box plot with mean and confidence interval
colors = ['#78C850', '#F08030', '#6890F0', '#F8D030', '#F85888', '#705898', '#98D8D8']
# ax = sns.boxplot(data=data, palette=colors, showmeans=True, fliersize=0.5,
#                 meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
#                 notch=True, width=0.2, boxprops=dict(alpha=1.0))
active0 = pd.read_csv("traffic-profile/active-0.csv")
active1 = pd.read_csv("traffic-profile/active-1.csv")

X0 = active0["Step"]
Y0 = active0["active-0"]
MV0 = active0["active-0"].rolling(50).mean()

X1 = active1["Step"]
Y1 = active1["active-1"]
MV1 = active1["active-1"].rolling(50).mean()

for val in active0["Step"].to_numpy():
    index = val//100000
    lamda0.append(init_lambda0[index])

for val in active1["Step"].to_numpy():
    index = val//100000
    lamda1.append(init_lambda1[index])

import numpy as np

X = X0.to_numpy()

samples = 3000
print(X.shape)
indices = np.random.choice(X.shape[0], samples, replace=False)

X = X[indices]
sns.lineplot(x=X, y=Y0.to_numpy()[indices])
sns.lineplot(x=X, y=MV0.to_numpy()[indices])
sns.lineplot(x=X, y=np.array(lamda0)[indices])

# plt.plot(active1["Step"], active1["active-1"])
import tikzplotlib

# plt.legend()
tikzplotlib.save("active0.tex")
plt.show()
print(len(active0))
print(len(active1))
