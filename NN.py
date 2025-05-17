"""
This file contains the architecture of a two-headed Neural Network that has a policy and a value head.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class NeuralNet(nn.Module):

    def __init__(self):

        super(NeuralNet, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(14, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)

        self.policy_1 = nn.Linear(256, 64)
        self.policy_2 = nn.Linear(64, 6)

        self.value_1 = nn.Linear(256, 64)
        self.value_2 = nn.Linear(64, 1)

    
    def forward(self, x):
        # Shared layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # Policy head of the NN 
        policy = self.policy_1(x)
        policy = self.policy_2(policy)
        policy = F.softmax(policy, dim=1)

        # Value head of the NN 
        value = F.relu(self.value_1(x))
        value = torch.tanh(self.value_2(value))

        return policy, value


if __name__ == "__main__":

    model = NeuralNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    summary(model, input_size=(14,))

