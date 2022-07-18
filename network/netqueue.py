from queue import Queue
from .packet import Packet
from .globsim import SimGlobals
from collections import deque


class NetQueue():
    def __init__(self, maxsize=1500):
        self.queue = deque(maxlen=maxsize)
        self.allocated_resources = 1

        # perm stats
        self.perm_total_enqueued = 0
        self.perm_total_served = 0
        self.perm_total_dropped = 0

        # temp stats
        self.temp_total_enqueued = 0
        self.temp_total_served = 0
        self.temp_total_dropped = 0

        self.stats = []

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

    def step(self):
        self.update_dead_packets()
        available_bandwidth = SimGlobals.NET_TIMESLOT_DURATION * 0.001 * self.allocated_resources * SimGlobals.BANDWIDTH_PER_RESOURCE
        count = 0
        while available_bandwidth > 0 and not self.empty():
            count += 1
            self.temp_total_served +=1
            self.perm_total_served += 1
            served_packet = self.get()
            available_bandwidth -= served_packet.size
            served_packet.served_at = SimGlobals.NET_TIMESLOT_STEP
            SimGlobals.served_packets.append(served_packet)

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
            (SimGlobals.NET_TIMESLOT_STEP, self.temp_total_enqueued, self.temp_total_served, self.temp_total_dropped, len(self.queue)))

        self.temp_total_enqueued = 0
        self.temp_total_served = 0
        self.temp_total_dropped = 0

    def reset_perm_stats(self):
        self.perm_total_enqueued = 0
        self.perm_total_served = 0
        self.perm_total_dropped = 0


    def update_dead_packets(self):
        for i in range(len(self.queue)):
            if self.queue[i].is_dead():
                SimGlobals.dropped_after_enqueue_packets.append(self.queue[i])
                del self.queue[i]
