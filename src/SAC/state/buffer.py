import numpy as np
import torch
from collections import deque
import random
import pickle


class ReplayBuffer():
    def __init__(self, buffer_limit, state_dim, action_dim, DEVICE):
        self.buffer = deque(maxlen=buffer_limit)
        self.dev = DEVICE
        self.state_dim = state_dim
        self.action_dim = action_dim

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_batch = torch.empty((n, self.state_dim), dtype=torch.float)
        a_batch = torch.empty((n, self.action_dim), dtype=torch.float)
        r_batch = torch.empty((n, 1), dtype=torch.float)
        s_next_batch = torch.empty((n, self.state_dim), dtype=torch.float)
        d_batch = torch.empty((n, 1), dtype=torch.float)

        for i, transition in enumerate(mini_batch):
            s, a, r, s_, d = transition
            s_batch[i] = torch.tensor(s, dtype=torch.float)
            a_batch[i] = torch.tensor(a, dtype=torch.float)
            r_batch[i] = torch.tensor(r, dtype=torch.float)
            s_next_batch[i] = torch.tensor(s_, dtype=torch.float)
            d_batch[i] = 0.0 if d else 1.0

        return s_batch.to(self.dev), a_batch.to(self.dev), r_batch.to(self.dev), s_next_batch.to(self.dev), d_batch.to(
            self.dev)

    def size(self):
        return len(self.buffer)

    def save(self, path):
        pickle.dump(self.buffer, open(path + '/replaybuffer.pkl', 'wb'))

    def load(self, path):
        self.buffer = pickle.load(open(path + 'replaybuffer.pkl', 'rb'))
