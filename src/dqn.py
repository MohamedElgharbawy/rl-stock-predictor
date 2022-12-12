from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, state_size, action_size, fc1_size=64, fc2_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, action_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.tanh(x)
