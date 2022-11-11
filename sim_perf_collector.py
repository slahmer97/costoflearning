import csv


class PerfCollector:
    def __init__(self, filename="xd"):
        self.file = open('sim-res/{}.csv'.format(filename), 'w')
        header = ["step", "slice0:throughputs", "slice1:throughputs", "slice0:drop_rates","slice1:drop_rates",
                  "slice0:death_rates","slice1:death_rates", "slice0:queue_sizes", "slice1:queue_sizes",
                  "slice0:urgent_packets", "slice1:urgent_packets", "slice0:latency_per_packet",
                  "slice1:latency_per_packet", "learner:throughputs", "learner:queue_sizes"
                  ]
        self.writer = csv.DictWriter(self.file, fieldnames=header)
        self.writer.writeheader()
        self.step = 0
        self.stats = {
            "step": [],
            "slice0:throughputs": [],
            "slice1:throughputs": [],

            "slice0:drop_rates": [],
            "slice1:drop_rates": [],

            "slice0:death_rates": [],
            "slice1:death_rates": [],

            "slice0:queue_sizes": [],
            "slice1:queue_sizes": [],

            "slice0:urgent_packets": [],
            "slice1:urgent_packets": [],

            "slice0:latency_per_packet": [],
            "slice1:latency_per_packet": [],

            "learner:throughputs": [],

        }
        self.episodic_stats = {
            "slice0:throughputs": 0,
            "slice1:throughputs": 0,

            "slice0:drop_rates": 0,
            "slice1:drop_rates": 0,

            "slice0:death_rates": 0,
            "slice1:death_rates": 0,

            "slice0:queue_sizes": 0,
            "slice1:queue_sizes": 0,

            "slice0:urgent_packets": 0,
            "slice1:urgent_packets": 0,

            "slice0:latency_per_packet": 0,
            "slice1:latency_per_packet": 0,

            "learner:throughputs": 0,
            "learner:queue_sizes": 0,
        }

    def push_stats(self, **kwargs):
        self.step += 1
        self.episodic_stats["slice0:throughputs"] += kwargs["s0throughputs"]
        self.episodic_stats["slice1:throughputs"] += kwargs["s1throughputs"]

        self.episodic_stats["slice0:drop_rates"] += kwargs["s0drop_rates"]
        self.episodic_stats["slice1:drop_rates"] += kwargs["s1drop_rates"]

        self.episodic_stats["slice0:death_rates"] += kwargs["s0death_rates"]
        self.episodic_stats["slice1:death_rates"] += kwargs["s1death_rates"]

        self.episodic_stats["slice0:queue_sizes"] += kwargs["s0queue_sizes"]
        self.episodic_stats["slice1:queue_sizes"] += kwargs["s1queue_sizes"]

        self.episodic_stats["slice0:urgent_packets"] += kwargs["s0urgent_packets"]
        self.episodic_stats["slice1:urgent_packets"] += kwargs["s1urgent_packets"]

        self.episodic_stats["slice0:latency_per_packet"] += kwargs["s0latency_per_packet"]
        self.episodic_stats["slice1:latency_per_packet"] += kwargs["s1latency_per_packet"]

        self.episodic_stats["learner:throughputs"] += kwargs["lthroughputs"]
        self.episodic_stats["learner:queue_sizes"] += kwargs["lqueue_sizes"]

        if self.step % 1000 == 0:
            self.episodic_stats["step"] = self.step % 1000

            self.episodic_stats["slice0:queue_sizes"] = self.episodic_stats["slice0:queue_sizes"] / 1000.0
            self.episodic_stats["slice1:queue_sizes"] = self.episodic_stats["slice1:queue_sizes"] / 1000.0

            self.episodic_stats["slice0:urgent_packets"] = self.episodic_stats["slice0:urgent_packets"] / 1000.0
            self.episodic_stats["slice1:urgent_packets"] = self.episodic_stats["slice1:urgent_packets"] / 1000.0

            self.episodic_stats["slice0:latency_per_packet"] = self.episodic_stats["slice0:latency_per_packet"] / 1000.0
            self.episodic_stats["slice1:latency_per_packet"] = self.episodic_stats["slice1:latency_per_packet"] / 1000.0

            self.episodic_stats["learner:queue_sizes"] = self.episodic_stats["learner:queue_sizes"] / 1000.0

            self.writer.writerow(self.episodic_stats)
            self.file.flush()
            self.episodic_stats = {
                "slice0:throughputs": 0,
                "slice1:throughputs": 0,

                "slice0:drop_rates": 0,
                "slice1:drop_rates": 0,

                "slice0:death_rates": 0,
                "slice1:death_rates": 0,

                "slice0:queue_sizes": 0,
                "slice1:queue_sizes": 0,

                "slice0:urgent_packets": 0,
                "slice1:urgent_packets": 0,

                "slice0:latency_per_packet": 0,
                "slice1:latency_per_packet": 0,

                "learner:throughputs": 0,
                "learner:queue_sizes": 0,
            }
