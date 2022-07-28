from collections import deque

from network.ActiveUserModel import ActiveUsers
from network.flow_gen import OnOffFlowGenerator
from network.netqueue import NetQueue
from network.network import Network
from network.packet import Packet
from matplotlib import pyplot as plt
import numpy as np

c = ActiveUsers(max_users=7)
c.get_mean()

c.simulate()