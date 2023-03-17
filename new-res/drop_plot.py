import numpy as np
import pandas
import pandas as pd
from matplotlib import pyplot as plt

task_id = "00"
data = pd.read_csv("task-{}/drop0.csv".format(task_id)).tail(100000)
size = 200000
ofb = pandas.DataFrame(
    {
        "packet_loss": np.random.choice(data["{}-ofb".format(task_id)].to_numpy().reshape(-1), size=size),
        "strategy": ["ofb"] * size
    }
)
print(len(ofb))
dynamic = pandas.DataFrame(
    {
        "packet_loss": np.random.choice(data["{}-dynamic".format(task_id)].to_numpy().reshape(-1), size=size)
        ,
        "strategy": ["dynamic"] * size
    }
)
print(len(dynamic))

always1 = pandas.DataFrame(
    {
        "packet_loss": np.random.choice(data["{}-static-1".format(task_id)].to_numpy().reshape(-1), size=size),
        "strategy": ["static-1"] * size
    }
)
print(len(always1))

every10 = pandas.DataFrame(
    {
        "packet_loss": np.random.choice(data["{}-every-10".format(task_id)].to_numpy().reshape(-1), size=size),
        "strategy": ["every-10ms"] * size
    }
)
print(len(every10))

every100 = pandas.DataFrame(
    {
        "packet_loss": np.random.choice(data["{}-every-100".format(task_id)].to_numpy().reshape(-1), size=size),
        "strategy": ["every-100ms"] * size
    }
)
print(len(every100))

# data = pd.DataFrame({
#    "ofb": ofb,
#    "dynamic":dynamic,
#    "static-1": always1,
#    "every-10ms": every10,
#    "every-100ms": every100
# })

ofb = ofb[~ofb['packet_loss'].isna()]
dynamic = dynamic[~dynamic['packet_loss'].isna()]
always1 = always1[~always1['packet_loss'].isna()]
every10 = every10[~every10['packet_loss'].isna()]
every100 = every100[~every100['packet_loss'].isna()]

data = pd.concat([ofb, dynamic, always1, every10, every100])

# data = np.swapaxes(data, 0, 1)
print(data.tail(1000))
# plt.plot(steps, ofb, label="ofb")
# plt.plot(steps, dynamic, label="dynamic")
# plt.plot(steps, every100, label="every100")
# plt.plot(steps, every10, label="every10")
# plt.plot(steps, always1, label="always1")

import seaborn as sns

# Set the style of the plot
sns.set_style("whitegrid")
sns.set_context('paper')

# Create a box plot with mean and confidence interval
colors = ['#78C850', '#F08030', '#6890F0', '#F8D030', '#F85888', '#705898', '#98D8D8']
# ax = sns.boxplot(data=data, palette=colors, showmeans=True, fliersize=0.5,
#                 meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
#                 notch=True, width=0.2, boxprops=dict(alpha=1.0))

ax = sns.barplot(data=data, x="strategy", y="packet_loss", ci=99.999, palette=colors, dodge=False)

# ax = sns.ecdfplot(data=ofb, x="packet_loss", label="ofb")
# ax = sns.ecdfplot(data=dynamic, x="packet_loss", label="dynamic")
# ax = sns.ecdfplot(data=always1, x="packet_loss", label="static-1")
# ax = sns.ecdfplot(data=every10, x="packet_loss", label="every-10ms")
# ax = sns.ecdfplot(data=every100, x="packet_loss", label="every-100ms")

# ax = sns.barplot(data=data, x="strategy", y="perfs", ci=99.999, palette=colors, dodge=False)

# Set the labels and title
# ax.set(ylabel='$\Phi$')
# ax.set(xlabel=None)
# ax.legend(["1", "2", "3", "4", "5"])
# sns.despine(trim=True)
import tikzplotlib

# plt.legend()
tikzplotlib.save("packet-loss-task{}.tex".format(task_id))
plt.show()
