import pandas as pd
import glob
files = glob.glob("./*.csv")
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X, Y, Z = [], [], []

for file in files:
    data = pd.read_csv(file, delimiter=" ")
    trained, tested = file[2:-4].split("-")
    tlamda0, tlamda1 = [float(val) for val in trained.split(" ")]
    lamda0, lamda1 = [float(val) for val in tested.split(" ")]

    print(lamda0, lamda1, (data["reward"].mean() + 1000) / 1000.0)
    X.append(lamda0)
    Y.append(lamda1)
    Z.append((data["reward"] + 1000).mean() / 1000.0)
ax.scatter(xs=X, ys=Y, zs=Z)
#plt.show()
ax.set_xlabel('$\lambda_{0}$', fontsize=10)
ax.set_ylabel('$\lambda_{1}$', fontsize=10)
ax.set_zlabel('$\Phi()$', fontsize=10)

for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(0.1)