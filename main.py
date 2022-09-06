import numpy as np

from network.network import Network
from matplotlib import pyplot as plt
from stable_baselines3 import DQN

net = Network()

model = DQN("MlpPolicy", net, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)

print(net.action_space)
print(net.observation_space)

indices = []
cum_reward = []
mean_reward = []
for i in range(1000):
    s = net.reset()
    #print(s)
    indices.append(i + 1)
    r = 0
    X = [0]
    Y1 = [s[2]]
    Y2 = [s[3]]
    for step in range(100):
        a, states_ = model.predict(s)
        s, reward, done, info = net.step(a)
        #print("\t",s)

        r += reward
        X.append(step+1)
        Y1.append(s[2])
        Y2.append(s[3])
    cum_reward.append(r)
    mean_reward.append(np.mean(cum_reward[-min(60, len(cum_reward)): ]))
    print("step: {} -- mean-reward: {}".format(i, np.mean(cum_reward[-min(60, len(cum_reward)): ])))
    if i == 50:
        plt.plot(X, Y1, label='1')
        plt.plot(X, Y2, label='2')

        plt.legend()
        plt.show()
    # print(s,stats)
    # print(s)

#plt.plot(indices, mean_reward, label='cum-reward')
#plt.legend()
#plt.show()
