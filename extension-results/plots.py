import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set()


def plot_average_queue_sizes(files, queue=1):
    X = np.arange(0, 2000)
    print(X)
    for i, file in enumerate(files):
        currentFileDF = pd.read_csv("{}.csv".format(file))
        avg_queue_sizes = currentFileDF["slice{}:queue_sizes".format(queue)].rolling(50).mean().to_numpy()
        sns.lineplot(x=X, y=avg_queue_sizes, sort=False, lw=1)
        print(len(avg_queue_sizes))
    plt.legend(labels=['dynamic', 'out of band', 'static allocation (1 blocks)'])
    plt.xlabel("Time (slot)")
    plt.ylabel("Queue size (pkt)")
    # plt.savefig("figures/queue{}Size.png".format(queue + 1))
    plt.show()


def plot_average_drop_rate(files, queue=0):
    if queue == 0:
        metric = "slice0:drop_rates"
    elif queue == 1:
        metric = "slice1:death_rates"

    X = np.arange(0, 2000)
    for i, file in enumerate(files):
        currentFileDF = pd.read_csv("{}.csv".format(file))
        avg_drop_rate = currentFileDF[metric].ewm(span=50,
                                                  adjust=True).mean().to_numpy()
        sns.lineplot(x=X, y=avg_drop_rate, sort=False, lw=1)
    plt.legend(labels=['this work', 'infinite learning bandwidth', 'static allocation (1 blocks)', 'static allocation '
                                                                                                   '(2 block)'])
    plt.ylabel("Drop rate (pkt/s)")
    plt.xlabel("Time (s)")
    plt.show()


def plot_average_latency(files, queue=0):
    if queue == 0:
        metric = "slice0:latency_per_packet"
    elif queue == 1:
        metric = "slice1:latency_per_packet"
    X = np.arange(0, 2000)
    print(len(X))
    for i, file in enumerate(files):
        currentFileDF = pd.read_csv("{}.csv".format(file))
        avg_latency = currentFileDF[metric].ewm(span=200,
                                                  adjust=True).mean().to_numpy()
        for (XX, YY) in zip(X, avg_latency):
            print("{} {}".format(XX, YY))
        sns.lineplot(x=X, y=avg_latency, sort=False, lw=1)

    plt.legend(labels=['this work', 'infinite learning bandwidth', 'static allocation (1 blocks)', 'static allocation '
                                                                                                   '(2 block)'])
    plt.ylabel("Average Latency (ms)")
    plt.xlabel("Time (s)")
    plt.savefig("latency{}.png".format(queue))
    plt.show()


all_files = ["dynamic", "ofb", "static1"]
# plot_agent_reward_bins(files=["0.csv", "1.csv", "2.csv", "3.csv"])
#plot_average_queue_sizes(files=all_files, queue=1)
# plot_average_learner_throughputs(["0.csv"])


# Done
#plot_average_latency(all_files, queue=0)

plot_average_drop_rate(files=all_files, queue=0)
