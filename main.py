import numpy as np
from network.network import Network
from matplotlib import pyplot as plt
from stable_baselines3 import DQN

net = Network()


model = DQN("MlpPolicy", net, verbose=1)

print(net.action_space)
print(net.observation_space)

indices = []
cum_reward = []
mean_reward = []
episode_len = 500
episodes = 100000
print('here')
for i in range(episodes):
    s = net.reset()
    # print(s)
    indices.append(i + 1)
    r = 0
    X = [0]
    Y1 = [s[2]]
    Y2 = [s[3]]
    for step in range(episode_len):
        a, states_ = model.predict(s)
        if step == episode_len - 1:
            net.set_end()

        s, reward, done, info = net.step(a)
        # print("\t",s)

        r += reward
        X.append(step + 1)
        Y1.append(s[2])
        Y2.append(s[3])
    cum_reward.append(r)
    mean_reward.append(np.mean(cum_reward[-min(60, len(cum_reward)):]))
    print("step: {} -- mean-reward: {}".format(i, np.mean(cum_reward[-min(60, len(cum_reward)):])))
    print("\t action0: {} -- action1: {} -- action2: {}".format(net.action0, net.action1, net.action2))
    if i in list(range(50, episodes, 50)):
        plt.plot(X, Y1, label='queue-1')
        plt.plot(X, Y2, label='queue-2')
        plt.legend()
        # plt.savefig("figures/simple-reward/q2^2.png")
        plt.show()
    # print(s,stats)
    # print(s)

# plt.plot(indices, mean_reward, label='cum-reward')
# plt.legend()
# plt.show()
