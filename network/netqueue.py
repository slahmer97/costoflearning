import queue
from queue import Queue
from .packet import Packet
from .globsim import SimGlobals
from collections import deque
from network.globsim import SimGlobals as G


class NetQueue:
    def __init__(self, type, maxsize=1500):
        self.queue = deque(maxlen=maxsize)
        self.allocated_resources = None

        self.type = type

        # perm stats
        self.perm_total_enqueued = 0
        self.perm_total_served = 0
        self.perm_total_dropped = 0
        self.perm_dead = 0
        # temp stats
        self.temp_total_enqueued = 0
        self.temp_total_served = 0
        self.temp_total_dropped = 0
        self.temp_dead = 0

        self.stats = []

    def allocate_resource(self, resources):
        self.allocated_resources = resources

    def empty(self) -> bool:
        return True if len(self.queue) == 0 else False

    def full(self) -> bool:
        return True if len(self.queue) == self.queue.maxlen else False

    def front(self) -> Packet:
        return self.queue[-1]

    def rear(self) -> Packet:
        return self.queue[0]

    def put(self, x: Packet) -> None:
        self.queue.append(x)

    def get(self) -> Packet:
        return self.queue.popleft()

    def __len__(self):
        return len(self.queue)

    def step(self):
        # self.update_dead_packets()
        available_bandwidth = SimGlobals.NET_TIMESLOT_DURATION_S * self.allocated_resources * SimGlobals.BANDWIDTH_PER_RESOURCE
        count = 0
        while available_bandwidth > 0 and not self.empty():
            count += 1
            self.temp_total_served += 1
            self.perm_total_served += 1
            served_packet = self.get()
            available_bandwidth -= served_packet.size
            served_packet.served_at = SimGlobals.NET_TIMESLOT_STEP
        return count

    def enqueue(self, packet):
        if isinstance(packet, Packet):
            return self.enqueue_packet_element(packet)
        elif isinstance(packet, list):
            return self.enqueue_packet_list(packet)
        else:
            raise "trying to enqueue a non packet element or list"

    def enqueue_packet_element(self, packet: Packet):
        if self.full():
            self.perm_total_dropped += 1
            self.temp_total_dropped += 1
            return 0
        packet.generated_at = SimGlobals.NET_TIMESLOT_STEP
        self.put(packet)
        self.perm_total_enqueued += 1
        self.temp_total_enqueued += 1

        return 1

    def enqueue_packet_list(self, packet_list: list):
        counter = 0
        for packet in packet_list:
            counter += self.enqueue_packet_element(packet)

        return counter

    def reset_temp_stats(self):
        self.stats.append(
            (SimGlobals.NET_TIMESLOT_STEP, self.temp_total_enqueued, self.temp_total_served, self.temp_total_dropped,
             len(self.queue)))

        self.temp_total_enqueued = 0
        self.temp_total_served = 0
        self.temp_total_dropped = 0
        self.temp_dead = 0

    def reset_perm_stats(self):
        self.perm_total_enqueued = 0
        self.perm_total_served = 0
        self.perm_total_dropped = 0
        self.perm_dead = 0

    def get_state(self):
        return [self.temp_total_enqueued, self.temp_total_dropped, self.temp_total_served, self.temp_dead,
                self.perm_total_enqueued, self.perm_total_dropped, self.perm_total_served, self.perm_dead]

    def update_dead_packets(self):
        indices_tobe_deleted = []

        for i in range(len(self.queue)):
            if self.queue[i].is_dead():
                self.perm_dead += 1
                self.temp_dead += 1
                indices_tobe_deleted.append(i)
        for i in sorted(indices_tobe_deleted, reverse=True):
            del (self.queue[i])

    def can_interrupt_next(self):
        interrupt = True
        non_feasible_packets = 0

        for i in range(len(self.queue)):
            packet_state, state_value = self.queue[i].get_packet_state()
            if (packet_state == "soft" or packet_state == "hard") and state_value <= 1:
                interrupt = True
                non_feasible_packets += 1

        return interrupt, non_feasible_packets


class ExperienceQueue:
    def __init__(self, init=0):
        self.max_size = 1500
        self.queue = deque()

        self.lifo = queue.LifoQueue(maxsize=1500)

        self.allocated_resources = init

        self.total_available_so_far = 0
        self.dropped = 0

        self.last_resource_usage = init

    def push(self, sample):
        self.lifo.put(sample)

        # if len(self.queue) <= self.max_size:
        #    self.queue.append(sample)
        # else:
        #    self.dropped += 1

    def step(self, additional_resources=0):

        self.last_resource_usage = self.allocated_resources + additional_resources
        assert additional_resources >= 0
        samples = []
        if self.lifo.qsize() < 1:
            return samples

        available_bandwidth = SimGlobals.NET_TIMESLOT_DURATION_S * (
                self.allocated_resources + additional_resources) * SimGlobals.BANDWIDTH_PER_RESOURCE

        self.total_available_so_far += available_bandwidth

        # print('Available bandwidth is: {}'.format(self.total_available_so_far))

        while self.total_available_so_far >= G.EXPERIENCE_SIZE and self.lifo.qsize() > 0:
            sample = self.lifo.get()  # self.queue.popleft()
            self.total_available_so_far -= G.EXPERIENCE_SIZE
            samples.append(sample)

        if self.lifo.qsize() == 0:
            self.total_available_so_far = 0

        return samples
