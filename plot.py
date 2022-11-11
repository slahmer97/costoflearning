import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set()


def errorfill(x, y, yerr, color="blue", alpha_fill=0.1, ax=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def plot_agent_reward_bins(files, bin=500):
    bins = np.arange(0, 2000 + 1, bin)
    bins = [0, 200, 400, 600, 2000]
    print(bins)
    X = bins[1:]
    bars = ["0-200", "200-400", "400-600", "600-2000k"]
    colors = ["b", "g", "r", "c"]
    barWidth = 0.2

    ind = np.arange(len(X))
    fake = [ind - 2 * barWidth, ind - barWidth, ind, ind + barWidth]
    labels = ["this work", "infinite learning bandwidth", "static allocation (1 block)", "static allocation (2 blocks)"]
    for i, file in enumerate(files):
        Y = []
        yErrors = []
        currentFileDF = pd.read_csv("sim-res/{}".format(file))
        groups = currentFileDF.groupby(pd.cut(currentFileDF.index, bins))

        for (name, group) in groups:
            stats = group["slice1:throughputs"].to_numpy()
            mean_ = np.mean(stats * 512)
            std = np.std(stats)
            Y.append(mean_)
            # yErrors.append(std*0.5)
        plt.bar(fake[i], Y, barWidth, label=labels[i])
        # errorfill(x=X, y=np.array(Y), color=colors[i], yerr=np.array(yErrors)*0.5)
    plt.xticks([0, 1, 2, 3],
               ["0-200K", "200K-400K", "400K-600K", "600K-2M"])
    plt.legend()
    plt.xlabel('steps')
    plt.ylabel('average death rate')
    plt.savefig("averageThroughputs.png")
    plt.show()
    # sea.barplot(data=currentFileDF, x=pd.cut(currentFileDF.index, bins), y="slice1:latency_per_packet")
    # plt.xticks(rotation=45)

    # plt.show()
    # print(groups)


# , "2.csv", "3.csv"


def plot_average_queue_sizes(files, queue=1):
    X = np.arange(0, 2000) * 1000
    print(X)
    for i, file in enumerate(files):
        currentFileDF = pd.read_csv("sim-res/{}".format(file))
        avg_queue_sizes = currentFileDF["slice{}:queue_sizes".format(queue)].ewm(span=50,
                                                                                 adjust=True).mean().to_numpy() * 1500.0
        sns.lineplot(x=X, y=avg_queue_sizes, sort=False, lw=1)
        print(len(avg_queue_sizes))
    plt.legend(labels=['this work', 'infinite learning bandwidth', 'static allocation (2 blocks)', 'static allocation '
                                                                                                   '(1 block)'])
    plt.xlabel("Time (slot)")
    plt.ylabel("Queue size (pkt)")
    plt.savefig("figures/queue{}Size.png".format(queue + 1))
    plt.show()


def plot_average_learner_throughputs(files):
    X = np.arange(0, 2000)
    indices = np.arange(0, 2000, 10)

    for i, file in enumerate(files):
        currentFileDF = pd.read_csv("sim-res/{}".format(file))
        avg_learner_throughputs = currentFileDF["learner:throughputs"].to_numpy()
        sns.lineplot(x=np.take(X, indices), y=np.take(avg_learner_throughputs,indices), sort=False, lw=1)
        print(len(avg_learner_throughputs))

    plt.ylabel("Forwarded Experiences (experience/s)")
    plt.xlabel("Time (s)")

    import tikzplotlib
    tikzplotlib.save("figures/averageForwardedExperiences.tex")
    plt.savefig("figures/forwardedExperiences.png")
    plt.show()


def plot_average_drop_rate(files, queue=0):
    if queue == 0:
        metric = "slice0:drop_rates"
    elif queue == 1:
        metric = "slice1:death_rates"
    indices = np.arange(0, 2000, 10)

    X = np.arange(0, 2000) * 1000
    for i, file in enumerate(files):
        currentFileDF = pd.read_csv("sim-res/{}".format(file))
        avg_drop_rate = currentFileDF[metric].ewm(span=50,
                                                  adjust=True).mean().to_numpy()
        sns.lineplot(x=np.take(X, indices), y=np.take(avg_drop_rate,indices), sort=False, lw=1)
    plt.legend(labels=['this work', 'infinite learning bandwidth', 'static allocation (1 blocks)', 'static allocation '
                                                                                                   '(2 block)'])
    plt.ylabel("Drop rate (pkt/s)")
    plt.xlabel("Time (s)")
    import tikzplotlib
    #tikzplotlib.save("figures/averageDropRateQueue{}.tex".format(queue + 1))
    #plt.savefig("figures/averageDropRateQueue{}.png".format(queue + 1))
    plt.show()


def plot_average_latecy(files, queue=0):
    if queue == 0:
        metric = "slice0:latency_per_packet"
    elif queue == 1:
        metric = "slice1:latency_per_packet"
    indices = np.arange(0, 550, 10)
    print(indices)
    X = np.arange(0, 550)
    print(len(X))
    for i, file in enumerate(files):
        currentFileDF = pd.read_csv("sim-res/{}".format(file))
        avg_drop_rate = currentFileDF[metric].ewm(span=50,
                                                  adjust=True).mean().to_numpy()
        for (XX, YY) in zip(X, avg_drop_rate):
            print("{} {}".format(XX, YY))
        sns.lineplot(x=np.take(X, indices), y=np.take(avg_drop_rate,indices), sort=False, lw=1)

    plt.legend(labels=['this work', 'infinite learning bandwidth', 'static allocation (2 blocks)', 'static allocation '
                                                                                                   '(1 block)'])
    plt.ylabel("Average Latency (ms)")
    plt.xlabel("Time (s)")
    import tikzplotlib
    #tikzplotlib.save("figures/averageLatencyPerPacketQueue{}.tex".format(queue + 1))

    #plt.savefig("figures/averageLatencyPerPacketQueue{}.png".format(queue + 1))


    plt.show()


all_files = ["1.csv"]#, "1.csv", "2.csv", "3.csv"]
# plot_agent_reward_bins(files=["0.csv", "1.csv", "2.csv", "3.csv"])
# plot_average_queue_sizes(files=["0.csv", "1.csv", "2.csv", "3.csv"], queue=0)
#plot_average_learner_throughputs(["0.csv"])
#plot_average_drop_rate(files=["0.csv", "1.csv", "2.csv", "3.csv"], queue=1)
plot_average_latecy(all_files, queue=1)
