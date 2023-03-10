import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

task_id = "12"
data = pd.read_csv("task-{}/performance.csv".format(task_id))

steps = data["Step"].apply(lambda val: val if val <= 100000 else val * 4)
always1 = data["{}-static-1".format(task_id)].to_numpy().reshape(-1)
every10 = data["{}-every-10".format(task_id)].to_numpy().reshape(-1)
every100 = data["{}-every-100".format(task_id)].to_numpy().reshape(-1)
ofb = data["{}-ofb".format(task_id)].to_numpy().reshape(-1)
dynamic = data["{}-dynamic".format(task_id)].to_numpy().reshape(-1)

data = np.array([ofb, dynamic, always1, every10, every100])
data = np.swapaxes(data, 0, 1)
print(steps.head(100))
print(steps.tail(10))

# plt.plot(steps, ofb, label="ofb")
# plt.plot(steps, dynamic, label="dynamic")
# plt.plot(steps, every100, label="every100")
# plt.plot(steps, every10, label="every10")
# plt.plot(steps, always1, label="always1")

import seaborn as sns


# Set the style of the plot
sns.set_style("whitegrid")

# Create a box plot with mean and confidence interval
ax = sns.boxplot(data=data, palette="pastel", showmeans=True, meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"},
                   notch=True, width=0.6, boxprops=dict(alpha=.5))

# Set the labels and title
ax.set(ylabel='Values', title='Box Plot Example')

# Show the plot
sns.despine(trim=True)
plt.show()
