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
        reward_key = -1 if reward < 0 else 1
        if reward_key not in self.experience_dict:
            self.n_buffers += 1
            self.experience_dict[reward_key] = deque(maxlen=self.max_size)
        self.experience_dict[reward_key].append(experience)

    def __len__(self):
        return len(self.buffer)

    def full(self, percent=1):
        if self.n_buffers <= 0:
            return False

        for buffer in self.experience_dict.values():
            if len(buffer) < self.max_size * percent:
                return False
        return True

    def __repr__(self):
        s = ""
        for k in self.experience_dict.keys():
            s += "%s -> %s, " % (k, len(self.experience_dict[k]))
        return s

    # Equal sampling from each dict list of batch_size elements
    def sample(self, batch_size):
        assert(batch_size <= self.max_size and self.n_buffers > 0)
        actual_size = int(batch_size / self.n_buffers)
        lis = []
        for buffer in self.experience_dict.values():
            index = np.random.choice(np.arange(actual_size),
                                     size=actual_size,
                                     replace=False)
            for i in index:
                lis.append(buffer[i])

        np.random.shuffle(lis)
        return lis
