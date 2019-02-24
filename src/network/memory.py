from collections import deque
import numpy as np


class Memory():
    def __init__(self, max_size, reward_key):
        self.max_size = max_size
        self.reward_key = reward_key
        self.experience_dict = {}
        self.n_buffers = 0

    def add(self, experience):
        reward = experience[self.reward_key]
        if reward not in self.experience_dict:
            self.n_buffers += 1
            self.experience_dict[reward] = deque(maxlen=self.max_size)
        self.experience_dict[reward].append(experience)

    def __len__(self):
        return len(self.buffer)

    @property
    def full(self):
        if self.n_buffers <= 0:
            return False

        for buffer in self.experience_dict.values():
            if len(buffer) < self.max_size:
                return False
        return True

    # Equal sampling from each dict list of batch_size elements
    def sample(self, batch_size):
        assert(batch_size <= self.max_size and self.n_buffers > 0)
        actual_size = batch_size / self.n_buffers
        lis = []
        for buffer in self.experience_dict.values():
            index = np.random.choice(np.arange(actual_size),
                                     size=actual_size,
                                     replace=False)
            for i in index:
                lis.append(buffer[int(i)])

        return lis