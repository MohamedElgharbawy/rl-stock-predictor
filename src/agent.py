import numpy as np
import random

# Importing the model (function approximator for Q-table)
from dqn import DQN
from replay_buffer import ReplayBuffer

import torch
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    """Interacts with and learns form environment."""

    def __init__(self, stock_size, finance_size, action_size):
        """Initialize an Agent object.

        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        # Q- Network
        self.qnetwork_local = DQN(stock_size, finance_size, action_size).to(device)
        self.qnetwork_target = DQN(stock_size, finance_size, action_size).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

        # Initialize time step (for updating every UPDATE_EVERY step)
        self.t_step = 0

    def step(self, stock_state, finance_state, action, reward, next_stock_state, next_finance_state, done):
        # Save experience in replay memory
        self.memory.add(stock_state, finance_state, action, reward, next_stock_state, next_finance_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get radom subset and learn
            if len(self.memory) > BATCH_SIZE:
                experience = self.memory.sample()
                self.learn(experience, GAMMA)

    def act(self, stock_state, finance_state, eps=0):
        """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        stock_state = torch.from_numpy(stock_state).float().unsqueeze(0).to(device)
        finance_state = torch.from_numpy(finance_state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(stock_state, finance_state)
        self.qnetwork_local.train()

        # Epsilon -greedy action selction
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # Get random float from [-1, 1]
            return np.random.random() * 2 - 1

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        stock_state, finance_state, actions, rewards, next_stock_state, next_finance_state, done = experiences

        criterion = torch.nn.MSELoss()
        # Local model is one which we need to train so it's in training mode
        self.qnetwork_local.train()
        # Target model is one with which we need to get our target so it's in evaluation mode
        # So that when we do a forward pass with target model it does not calculate gradient.
        # We will update target model weights with soft_update function
        self.qnetwork_target.eval()
        # shape of output from the model (batch_size,action_dim) = (64,4)
        predicted_targets = self.qnetwork_local(stock_state, finance_state).gather(1, actions)

        with torch.no_grad():
            labels_next = self.qnetwork_target(next_stock_state, next_finance_state).detach().max(1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (gamma * labels_next * (1 - done))

        loss = criterion(predicted_targets, labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)