from torch import nn
from torch import cat
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, stock_size, finance_size, action_size, fc1_size=64, fc2_size=64):
        super(DQN, self).__init__()
        self.stock_nn = BaseNet(stock_size, action_size, fc1_size, fc2_size)
        self.finance_nn = BaseNet(finance_size, action_size, fc1_size, fc2_size)
        self.tanh = nn.Tanh()

    def forward(self, stock_state, finance_state):
        output1 = self.stock_nn(stock_state)
        output2 = self.finance_nn(finance_state)
        return self.tanh(cat((output1, output2), dim=1))

class BaseNet(nn.Module):
    def __init__(self, state_size, action_size, fc1_size=64, fc2_size=64):
        super(BaseNet, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
