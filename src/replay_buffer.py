from collections import deque, namedtuple
import random
import torch
import numpy as np

class ReplayBuffer:
    """Fixed -size buffe to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["stock_state",
                                                                 "finance_state",
                                                                 "action",
                                                                 "reward",
                                                                 "next_stock_state",
                                                                 "next_finance_state",
                                                                 "done"])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add(self, stock_state, finance_state, action, reward, next_stock_state, next_finance_state, done):
        """Add a new experience to memory."""
        e = self.experiences(stock_state, finance_state, action, reward, next_stock_state, next_finance_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        stock_states = torch.from_numpy(
            np.vstack([e.stock_state for e in experiences if e is not None])).float().to(self.device)
        finance_states = torch.from_numpy(
            np.vstack([e.finance_state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_stock_states = torch.from_numpy(
            np.vstack([e.next_stock_state for e in experiences if e is not None])).float().to(self.device)
        next_finance_states = torch.from_numpy(
            np.vstack([e.next_finance_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return stock_states, finance_states, actions, rewards, next_stock_states, next_finance_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)