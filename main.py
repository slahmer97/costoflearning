from collections import deque

from network.flow_gen import OnOffFlowGenerator
from network.netqueue import NetQueue
from network.network import Network
from network.packet import Packet
from matplotlib import pyplot as plt
import numpy as np
a = Network()

a.reset()

for i in range(5000):
    a.step()
    print("-----------------------------------------")



index = []
enqueue = []
served = []
dropped = []
sizes = []
for  (step, temp_total_enqueued, temp_total_served, temp_total_dropped, size) in a.slices[1].stats:
    index.append(step)
    enqueue.append(temp_total_enqueued)
    served.append(temp_total_served)
    dropped.append(temp_total_dropped)
    sizes.append(size)
print(sizes)

plt.plot(index, served)

plt.show()