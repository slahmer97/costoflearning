from os import walk
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def plotAll():
    paths = ["./gap-res/0-0/", "./gap-res/15-20/", "./gap-res/30-10/"]
    legends = ["training-at-each-points", "using-pi-15-20", "using-pi-30-10"]
    for path in paths:
        for (dirpath, dirnames, filenames) in walk(path):
            files = filenames

        indices = []
        perfs = []
        for filename in files:
            if filename[0] == "a":
                first, second = filename.split('  ')
            else:
                first, second = filename.split('  ')

            trained_on_u0, trained_on_u1 = [int(e) for e in
                                            first.replace("train-on=", "").replace("a", "").replace("-300", "").split(
                                                "-")]
            tested_on_u0, tested_on_u1 = [int(e) for e in
                                          second.replace("tested-on=", "").replace(".csv", "").split("-")]
            data = pd.read_csv("{}{}".format(path, filename))
            drop0 = data["slice0:drop_rates"].tail(50).to_numpy().reshape(-1)
            death1 = data["slice1:death_rates"].tail(50).to_numpy().reshape(-1)
            # indices.append( (17-tested_on_u1 * 0.5) / 15)
            indices.append((17-tested_on_u1 * 0.5) / 15)
            perf = - np.mean(0.25 * drop0 / tested_on_u0 + 0.75 * death1 / tested_on_u1)

            perfs.append(perf)

            print(trained_on_u0, trained_on_u1, tested_on_u0, tested_on_u1)

        plt.scatter(indices, perfs)
    plt.legend(legends)
    plt.savefig("perf.png")
    plt.show()



plotAll()
