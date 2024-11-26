import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(14, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)

        # Policy head
        self.policy_1 = nn.Linear(256, 64)
        self.policy_2 = nn.Linear(64, 6)

        # Value head
        self.value_1 = nn.Linear(256, 64)
        self.value_2 = nn.Linear(64, 1)

    
    def forward(self, x):
        # Shared layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # Policy head
        policy = self.policy_1(x)
        policy = self.policy_2(policy)
        policy = F.softmax(policy, dim=1)
        #policy = F.log_softmax(policy, dim=1) I was using this but after training all the probabilities became negative and clamping changed them to 0 so that sampling from it meant acting randomly

        # Value head
        value = F.relu(self.value_1(x))
        value = torch.tanh(self.value_2(value))

        return policy, value


"""
model = NeuralNet()
board_state = torch.tensor([4,4,4,4,4,4,0,4,4,4,4,4,4,0], dtype=torch.float)
print(model.summary())
#print(board_state)
board_state = board_state.unsqueeze(0)  
#print(board_state)
#print(len(board_state))
#print(len(board_state[0]))

probabilities, value = model(board_state)
#print(probabilities)
#print(value)
"""
