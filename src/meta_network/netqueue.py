import queue
from collections import deque
from src.meta_network.packet import Packet


class NetQueue:
    def __init__(self, sim, type, maxsize=1500):
        self.avg_latency = 0.0
        self.cum_cost = 0.0
        self.sim = sim
        self.queue = deque(maxlen=maxsize)
        self.allocated_resources = None

        self.type = type

        # perm stats
        self.perm_total_enqueued = 0
        self.perm_total_served = 0
        self.perm_total_dropped = 0
        self.perm_dead = 0
        self.perm_resent = 0
        # temp stats
        self.temp_total_enqueued = 0
        self.temp_total_served = 0
        self.temp_total_dropped = 0
        self.temp_dead = 0
        self.temp_resent = 0

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
        available_bandwidth = self.sim.NET_TIMESLOT_DURATION_S * self.allocated_resources * self.sim.BANDWIDTH_PER_RESOURCE
        count = 0
        while available_bandwidth > self.sim.NET_TIMESLOT_DURATION_S * self.sim.BANDWIDTH_PER_RESOURCE and not self.empty():
            count += 1

            self.temp_total_served += 1
            self.perm_total_served += 1
            served_packet = self.get()
            available_bandwidth -= served_packet.size
            served_packet.served_at = self.sim.NET_TIMESLOT_STEP

            current_latency = served_packet.served_at - served_packet.generated_at
            # if self.type == 1 and current_latency > 50:
            #    print("packet generated:{} -- served: {} -- latency: {}".format(served_packet.generated_at, served_packet.served_at, current_latency))

            if not self.sim.send_success():
                self.temp_resent += 1
                self.perm_resent += 1
                self.queue.appendleft(served_packet)
            else:
                self.avg_latency += current_latency
                val = self.sim.urllc_cost(current_latency)
                # if self.type == 1:
                #    print("\tlatency: {} -- val: {} -- total: {} -- resent: {}".format(current_latency, val, self.temp_total_served, self.temp_resent))
                self.cum_cost += val

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
        packet.generated_at = self.sim.NET_TIMESLOT_STEP
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

        # self.stats.append(
        #     (SimGlobals.NET_TIMESLOT_STEP, self.temp_total_enqueued, self.temp_total_served, self.temp_total_dropped,
        #      len(self.queue)))

        self.temp_total_enqueued = 0
        self.temp_total_served = 0
        self.temp_total_dropped = 0
        self.temp_dead = 0
        self.temp_resent = 0

        self.avg_latency = 0.0

        self.cum_cost = 0.0

    def reset_perm_stats(self):
        self.perm_total_enqueued = 0
        self.perm_total_served = 0
        self.perm_total_dropped = 0
        self.perm_dead = 0
        self.perm_resent = 0

    def get_state(self):
        divider = self.temp_total_served - self.temp_resent
        assert divider >= 0

        if divider == 0:
            avgLatency = 0.0
        else:
            avgLatency = self.avg_latency / float(divider)

        return [self.temp_total_enqueued, self.temp_total_dropped, self.temp_total_served, self.temp_dead,
                self.perm_total_enqueued, self.perm_total_dropped, self.perm_total_served, self.perm_dead,
                avgLatency, self.temp_resent, self.perm_resent, self.cum_cost]

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
        up = 0
        sp = 0
        for i in range(len(self.queue)):
            packet_state, hard_life_counter, soft_life_counter = self.queue[i].get_packet_state()
            if packet_state == "good":
                continue
            if packet_state == "hard":
                up += 1
            if packet_state == "soft":
                sp += 1

        return up, sp


class ExperienceQueue:
    def __init__(self, sim, init=0, queue_type="fifo"):
        self.max_size = 1500
        self._sim = sim
        self.queue_type = queue_type

        if queue_type == "fifo":
            self.queue = deque(maxlen=self.max_size)
        else:
            self.lifo = deque(maxlen=self.max_size)

        self.allocated_resources = init

        self.total_available_so_far = 0
        self.dropped = 0

        self.last_resource_usage = init

        print("[+] Experience Queue was created with init_res: {} -- type: {} -- max_size: {}".format(
            self.allocated_resources, self.queue_type, self.max_size))

        self.stop = False

    def push(self, sample, error):

        if self.queue_type == "fifo":
            if len(self.queue) == self.max_size:
                self.dropped += 1
            self.queue.append(sample)
        elif self.queue_type == "lifo":
            if len(self.lifo) == self.max_size:
                self.dropped += 1

            self.lifo.append(sample)
        else:
            raise Exception("unknown queue type")
        # if len(self.queue) <= self.max_size:
        #    self.queue.append(sample)
        # else:
        #    self.dropped += 1

    def __len__(self):
        if self.queue_type == "fifo":
            return len(self.queue)
        else:
            return len(self.lifo)

    def get(self):
        if self.queue_type == "fifo":
            return self.queue.popleft()
        else:
            return self.lifo.pop()

    def reset(self):
        if self.queue_type == "fifo":
            self.queue = deque(maxlen=self.max_size)
        else:
            self.lifo = deque(maxlen=self.max_size)

    def step(self, additional_resources=0):

        if self.stop:
            return []

        self.last_resource_usage = self.allocated_resources + additional_resources
        assert additional_resources >= 0
        samples = []
        if len(self) < 1:
            return samples

        available_bandwidth = self._sim.NET_TIMESLOT_DURATION_S * (
                self.allocated_resources + additional_resources) * self._sim.BANDWIDTH_PER_RESOURCE

        self.total_available_so_far += available_bandwidth

        # print('Available bandwidth is: {}'.format(self.total_available_so_far))

        while self.total_available_so_far >= self._sim.EXPERIENCE_SIZE and len(self) > 0:
            self.total_available_so_far -= self._sim.EXPERIENCE_SIZE

            if self._sim.send_success():
                sample = self.get()  # self.queue.popleft()
                samples.append(sample)

        if len(self) == 0:
            self.total_available_so_far = 0

        return samples



class ExperiencePriorityQueue:
    def __init__(self, sim, init=0, **kwargs):
        self.max_size = 1500
        self._sim = sim
        self.allocated_resources = init
        self.total_available_so_far = 0
        self.dropped = 0
        self.last_resource_usage = init
        self.queue = queue.PriorityQueue(maxsize=self.max_size)
        self.id = 0
        print("[+] Experience Queue was created with init_res: {} -- type: Priority Queue -- max_size: {}".format(
            self.allocated_resources, self.max_size))

        self.stop = False

    def push(self, sample, error):
        priority = -abs(error) if error is not None else 0.0
        self.id += 1
        assert isinstance(priority, float)
        if self.queue.qsize() < self.max_size:
            self.dropped += 1
            self.queue.put((priority, self.id,sample), block=False)
        else:
            s = self.queue.get()
            self.queue.put((priority, self.id, sample), block=False)

    def __len__(self):
        return self.queue.qsize()

    def get(self):
        return self.queue.get(block=False)

    def reset(self):
        self.queue = queue.PriorityQueue(maxsize=self.max_size)
        self.total_available_so_far = 0
        self.dropped = 0

    def step(self, additional_resources=0):

        if self.stop:
            return []

        self.last_resource_usage = self.allocated_resources + additional_resources
        assert additional_resources >= 0
        samples = []
        if len(self) < 1:
            return samples

        available_bandwidth = self._sim.NET_TIMESLOT_DURATION_S * (
                self.allocated_resources + additional_resources) * self._sim.BANDWIDTH_PER_RESOURCE

        self.total_available_so_far += available_bandwidth

        # print('Available bandwidth is: {}'.format(self.total_available_so_far))

        while self.total_available_so_far >= self._sim.EXPERIENCE_SIZE and len(self) > 0:
            self.total_available_so_far -= self._sim.EXPERIENCE_SIZE

            if self._sim.send_success():
                priority, id, sample = self.get()  # self.queue.popleft()
                samples.append(sample)

        if len(self) == 0:
            self.total_available_so_far = 0

        return samples
