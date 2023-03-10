import numpy as np
import pandas as pd
import glob

files = glob.glob("./logs/*.csv")
import matplotlib.pyplot as plt

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)

Perf0, Perf1, Y, X = [], [], [], []
Other = {}
for file in files:
    # print(file[7:-4])
    data = pd.read_csv(file)
    l0, l1, h = file[7:-4].split("-")

    lambda0 = int(l0) * 0.352112676
    lambda1 = int(l1) * 0.5


    def cost_latency2(pkt_latency):
        if pkt_latency < 50:
            return 0.0
        else:
            return - (1.0 / 70.0) * pkt_latency


    p0 = (-data["slice0:drop_rates"].mean() / lambda0 + 1000.0) / 1000.0
    p1 = ((-data["slice1:death_rates"].mean() - cost_latency2(
        data["slice1:latency_per_packet"].mean())) / lambda1 + 1000.0) / 1000.0
    # print(f"{l0}-{l1}\n\t{lambda0} {lambda1}\n\t{p0}")

    print(f"{lambda0} {lambda1} {p0} {file}")
    X.append(lambda0)
    Y.append(lambda1)
    Perf0.append(p0)
    Perf1.append(p1)
    Other[(lambda0, lambda1)] = (p0, p1)

# X = np.array(list(set(X)))
# Y = np.array(list(set(Y)))
# X, Y = np.meshgrid(X, Y)

# Perf0 = np.zeros(shape=X.shape)
# Perf1 = np.zeros(shape=X.shape)

# for i in range(X.shape[0]):
#    for j in range(X.shape[1]):
#        key = (X[i][j], Y[i][j])
#        Perf0[i][j] = Other[key][0]
#        Perf1[i][j] = Other[key][1]
Perf1 = np.array(Perf1)
Perf0 = np.array(Perf0)
from pylab import *

# colmap = cm.ScalarMappable(cmap=cm.hsv)
# colmap.set_array(Perf1)

# ax.scatter(X, Y, Perf1, c=cm.hsv(Perf1 / max(Perf1)), marker='o')
sp = ax.scatter(X, Y, c=Perf1, cmap="RdBu_r", marker='o')

# cb = fig.colorbar(colmap)
ax.set_xlabel('$\lambda_{0}$', fontsize=10)
ax.set_ylabel('$\lambda_{1}$', fontsize=10)
# ax.set_zlabel('$\Phi_{1}()$', fontsize=10)
fig.colorbar(sp, label="$\Phi_{1}$")

import tikzplotlib
tikzplotlib.save("perf1-2d.tex")
plt.show()
