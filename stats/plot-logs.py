import pandas as pd
import glob
files = glob.glob("./logs/*.csv")
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Perf0, Perf1, Y, X = [], [], [], []

for file in files:
    #print(file[7:-4])
    data = pd.read_csv(file)
    l0, l1, h = file[7:-4].split("-")
    lambda0 = int(l0) * 0.352112676
    lambda1 = int(l1) * 0.5

    p0 = (-data["slice0:drop_rates"].mean()/lambda0 + 1000.0) / 1000.0

    print(f"{l0}-{l1}\n\t{lambda0} {lambda1}\n\t{p0}")

    X.append(lambda0)
    Y.append(lambda1)
    Perf0.append(p0)

    #Z.append((data["reward"] + 1000).mean() / 1000.0)

ax.scatter(xs=X, ys=Y, zs=Perf0)
plt.show()
ax.set_xlabel('$\lambda_{0}$', fontsize=10)
ax.set_ylabel('$\lambda_{1}$', fontsize=10)
ax.set_zlabel('$\Phi()$', fontsize=10)

for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(0.1)
