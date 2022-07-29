from collections import deque

from network.ActiveUserModel import ActiveUsers
from network.flow_gen import OnOffFlowGenerator
from network.netqueue import NetQueue
from network.network import Network
from network.packet import Packet
from matplotlib import pyplot as plt
import numpy as np

net = Network()

s = net.reset()
print(s)
indices = [0]
active_users1 = [s[0]]
active_users2 = [s[1]]
bandwidth1 = [0]
bandwidth2 = [0]
queue1 = [s[2]]
queue2 = [s[3]]
for i in range(10000):
    indices.append(i+1)
    s, _, stats = net.step(1)
    #print(s)
    active_users1.append(s[0])
    active_users2.append(s[1])
    bandwidth1.append(stats[0])
    bandwidth2.append(stats[1])
    queue1.append(s[2])
    queue2.append(s[3])
plt.plot(indices, queue1, label='active_user1')
plt.plot(indices, queue2, label='active_user2')
plt.legend()
plt.show()