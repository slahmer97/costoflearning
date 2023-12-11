import numpy as np
import yaml


class Tasks:
    def __init__(self, environments):
        self.tasks = environments

        self.test_tasks = []

    def get_task(self, i=0):
        return self.tasks[i]

    def sample_task(self):
        return self.tasks[np.random.randint(0, len(self.tasks))]

    def sample_batch_tasks(self, batch=16):
        return self.tasks

    def sample_test_task(self):
        return self.test_tasks

    def dump_yaml(self):
        return yaml.dump(self.tasks)
