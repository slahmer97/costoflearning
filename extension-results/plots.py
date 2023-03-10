import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style("whitegrid")


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
        avg_drop_rate = currentFileDF[metric].ewm(span=5).mean().to_numpy()
        sns.lineplot(x=X * 1000, y=avg_drop_rate, sort=False, lw=1)
    plt.legend(labels=['this work', 'infinite learning bandwidth', 'static allocation (1 blocks)', 'static allocation '
                                                                                                   '(2 block)'])

    plt.ylabel("Drop rate (bytes/s)")
    plt.xlabel("Time (s)")

    import tikzplotlib
    tikzplotlib.save("DropRateQueue{}.tex".format(queue))
    plt.savefig("DropRateQueue{}.png".format(queue))
    # plt.show()
    plt.show()


def plot_performance(files):
    currentFileDF = pd.read_csv("reward.csv")
    x = currentFileDF["Step"].to_numpy()
    ofb = currentFileDF["ofb-final-steps2M-Seed1 - learner:performance"].ewm(span=1).mean().to_numpy()
    dynamic = currentFileDF["dynamic-final-steps2M-Seed1 - learner:performance"].ewm(span=1).mean().to_numpy()
    static1 = currentFileDF["static-final-steps2M-Seed1 - learner:performance"].ewm(span=1).mean().to_numpy()
    sns.lineplot(x=x, y=ofb, sort=False, lw=1)
    sns.lineplot(x=x, y=dynamic, sort=False, lw=1)
    sns.lineplot(x=x, y=static1, sort=False, lw=1)
    plt.legend(labels=['this work', 'infinite learning bandwidth', 'static allocation (1 blocks)', 'static allocation '
                                                                                                   '(2 block)'])

    import tikzplotlib
    tikzplotlib.save("perf.tex")
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
        avg_latency = currentFileDF[metric].ewm(span=1).mean().to_numpy()
        for (XX, YY) in zip(X, avg_latency):
            print("{} {}".format(XX, YY))
        sns.lineplot(x=X, y=avg_latency, sort=False, lw=1)

    plt.legend(labels=['this work', 'infinite learning bandwidth', 'static allocation (1 blocks)', 'static allocation '
                                                                                                   '(2 block)'])
    plt.ylabel("Average Latency (ms)")
    plt.xlabel("Time (s)")
    import tikzplotlib
    tikzplotlib.save("latency{}.tex".format(queue))
    plt.savefig("latency{}.png".format(queue))
    plt.show()


def incoming_traffic():
    traffic0 = pd.read_csv("incoming-traffic0.csv")
    traffic1 = pd.read_csv("incoming-traffic1.csv")

    traffic0 = traffic0[traffic0['ofb-final-steps2M-Seed1 - slice0:incoming-traffic'].notnull()].sample(n=2000)
    traffic1 = traffic1[traffic1['ofb-final-steps2M-Seed1 - slice1:incoming-traffic'].notnull()].sample(n=2000)

    X = traffic0['Step'].to_numpy().reshape(-1)
    Y0 = traffic0['ofb-final-steps2M-Seed1 - slice0:incoming-traffic'].ewm(span=5).mean().to_numpy().reshape(-1)
    sns.scatterplot(x=X, y=Y0)

    X = traffic1['Step'].to_numpy().reshape(-1)
    Y1 = traffic1['ofb-final-steps2M-Seed1 - slice1:incoming-traffic'].ewm(span=5).mean().to_numpy().reshape(-1)
    sns.scatterplot(x=X, y=Y1)
    import tikzplotlib
    tikzplotlib.save("traffic.tex")
    plt.savefig("traffic.png")
    plt.show()


def learning_plane_resources():
    currentFileDF = pd.read_csv("dynamic.csv")
    X = np.arange(0, 2000)
    resources = currentFileDF["learner:throughputs"].ewm(span=1).mean().to_numpy() * 512 * 8 / pow(10, 6)
    sns.lineplot(x=X, y=resources, sort=False, lw=1)
    plt.legend(labels=['this work'])

    import tikzplotlib
    tikzplotlib.save("learning_plane.tex")
    plt.show()


def averagePerformance():
    currentFileDF = pd.read_csv("reward.csv")

    x = currentFileDF["Step"].to_numpy()
    ofb = currentFileDF["ofb-final-steps2M-Seed1 - learner:performance"].ewm(span=1).mean().to_numpy()
    tmp = np.array_split(ofb, 4)

    dfOFB = pd.DataFrame({
        '[0-499]': tmp[0],
        '[500-999]': tmp[1],
        '[1000-1499]': tmp[2],
        '[1500-1999]': tmp[3]
    })
    dynamic = currentFileDF["dynamic-final-steps2M-Seed1 - learner:performance"].ewm(span=70).mean().to_numpy()
    tmp = np.array_split(dynamic, 4)

    dfDynamic = pd.DataFrame({
        '[0-499]': tmp[0],
        '[500-999]': tmp[1],
        '[1000-1499]': tmp[2],
        '[1500-1999]': tmp[3]
    })

    static1 = currentFileDF["static-final-steps2M-Seed1 - learner:performance"].ewm(span=1).mean().to_numpy()
    tmp = np.array_split(static1, 4)

    dfStatic = pd.DataFrame({
        '[0-499]': tmp[0],
        '[500-999]': tmp[1],
        '[1000-1499]': tmp[2],
        '[1500-1999]': tmp[3]
    })

    dfOFB = pd.melt(dfOFB)
    dfOFB['setup'] = 'OutOfBand'

    dfDynamic = pd.melt(dfDynamic)
    dfDynamic['setup'] = 'Dynamic'

    dfStatic = pd.melt(dfStatic)
    dfStatic['setup'] = 'Static'

    df = pd.concat([dfOFB, dfDynamic, dfStatic], axis=0)

    print(df.head())
    # sns.lineplot(x=x, y=dynamic, sort=False, lw=1)
    # sns.lineplot(x=x, y=static1, sort=False, lw=1)
    # plt.legend(labels=['this work', 'infinite learning bandwidth', 'static allocation (1 blocks)', 'static allocation '
    #                                                                                               '(2 block)'])
    sns.boxplot(x='variable', y='value', data=df, hue='setup', showfliers=False, palette='Set3')
    plt.xticks(rotation=45)
    # sns.stripplot(x='variable', y='value', data=df, hue='setup', dodge=True, jitter=True, color='black')
    # import tikzplotlib
    # tikzplotlib.save("AveragePerf.tex")
    plt.show()


def plotCDFdelay(files, queue=1):
    # sns.set_context("talk")
    if queue == 0:
        metric = "slice0:latency_per_packet"
    elif queue == 1:
        metric = "slice1:latency_per_packet"
    X = np.arange(0, 2000)
    print(len(X))
    #fig, axs = plt.subplots(1)
    # fig, axes = plt.subplots(2, 2)
    for i, file in enumerate(files):
        currentFileDF = pd.read_csv("{}.csv".format(file))
        avg_latency = currentFileDF[metric].to_numpy()  # .ewm(span=1).mean().to_numpy()
        tmp = np.array_split(avg_latency, 4)
        x = np.array_split(X, 4)
        #for iter, (XX, YY) in enumerate(zip(x, tmp)):
        #    print("{} {}".format(XX, YY))
        sns.ecdfplot(x=tmp[3])

    #for ax in axs.flat:
    #    ax.set(xlabel='delay (ms)', ylabel='CDF')
    #    ax.label_outer()

    #axs[0][0].set_title('[0-499]')
    #axs[0][1].set_title('[500-999]')
    #axs[1][0].set_title('[1000-1499]')
    #axs[1][1].set_title('[1500-2000]')
    #fig.legend(loc='upper center', labels=["Dynamic", "OutOfBand", "Static (1 block)"])

    #fig.legend(bbox_to_anchor=(1.3, 0.6),
    #           labels=['Dynamic', 'outOfBand', 'static allocation (1 blocks)'])
    #fig.tight_layout()
    # fig.ylabel("Average Latency (ms)")
    # plt.xlabel("Time (s)")
    import tikzplotlib
    tikzplotlib.save("cdf-latency1500-1999-{}.tex".format(queue))
    # plt.savefig("latency{}.png".format(queue))
    plt.show()


def plotDropRate(files, queue=1):
    if queue == 0:
        metric = "slice0:drop_rates"
    elif queue == 1:
        metric = "slice1:death_rates"

    tmp = [[], [], []]
    for i, file in enumerate(files):
        currentFileDF = pd.read_csv("{}.csv".format(file))
        avg_latency = currentFileDF[metric].to_numpy()
        t = np.array_split(avg_latency, 4)
        tmp[i].append(t)

    tmp = np.array(tmp).reshape((3, 4, -1))
    dfOFB = pd.DataFrame({
        '[0-499]': tmp[0][0],
        '[500-999]': tmp[0][1],
        '[1000-1499]': tmp[0][2],
        '[1500-1999]': tmp[0][3]
    })
    dfDynamic = pd.DataFrame({
        '[0-499]': tmp[1][0],
        '[500-999]': tmp[1][1],
        '[1000-1499]': tmp[1][2],
        '[1500-1999]': tmp[1][3]
    })

    dfStatic = pd.DataFrame({
        '[0-499]': tmp[2][0],
        '[500-999]': tmp[2][1],
        '[1000-1499]': tmp[2][2],
        '[1500-1999]': tmp[2][3]
    })

    dfOFB = pd.melt(dfOFB)
    dfOFB['setup'] = 'OutOfBand'

    dfDynamic = pd.melt(dfDynamic)
    dfDynamic['setup'] = 'Dynamic'

    dfStatic = pd.melt(dfStatic)
    dfStatic['setup'] = 'Static'
    df = pd.concat([dfOFB, dfDynamic, dfStatic], axis=0)
    print(df.head(2))
    # plt.legend(labels=['this work', 'infinite learning bandwidth', 'static allocation (1 blocks)', 'static allocation '
    sns.boxplot(x='variable', y='value', data=df, hue='setup', showfliers=False, palette='Set3')
    plt.xticks(rotation=45)
    # sns.stripplot(x='variable', y='value', data=df, hue='setup', dodge=True, jitter=True, color='black')
    # import tikzplotlib
    # tikzplotlib.save("AveragePerf.tex")
    # plt.ylabel("Average Latency (ms)")
    # plt.xlabel("Time (s)")
    # import tikzplotlib
    # tikzplotlib.save("latency{}.tex".format(queue))
    # plt.savefig("latency{}.png".format(queue))
    plt.show()


all_files = ["dynamic", "ofb", "static1"]
# plot_agent_reward_bins(files=["0.csv", "1.csv", "2.csv", "3.csv"])
# plot_average_queue_sizes(files=all_files, queue=1)
# plot_average_learner_throughputs(["0.csv"])


# Done
# plotDropRate(files=all_files, queue=0)
# plot_average_drop_rate(files=all_files, queue=0)

plotCDFdelay(all_files)
