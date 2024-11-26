import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from NN import NeuralNet
import time

start_time = time.time()

file_path = r'C:\Users\erdur\Desktop\Using Neural Networks and MCTS to play Mancala\games\fully_random_games2.json'
model_path = r'C:\Users\erdur\Desktop\Using Neural Networks and MCTS to play Mancala\models\model_trained_on_random_games.pth'

# Loading the data
with open(file_path, 'r') as file:
    data = json.load(file)

def get_shape(lst):
    if isinstance(lst, list) and lst:
        return [len(lst)] + get_shape(lst[0])  # Get the length of the current list and recurse for the first element
    else:
        return []

print(get_shape(data))

game_states = []
action_probabilities = []
win_probabilities = []

for state in data:
    game_states.append(state[0])
    action_probabilities.append(state[1])
    win_probabilities.append(state[2])


N = 24 # The normalization value
game_states = [[min(N, stones)/ N for stones in state] for state in game_states]

# Round action probabilities to 3 decimal places
for actions in action_probabilities:
    for probability in actions:
        probability = round(probability, 3)


game_states = torch.tensor(game_states, dtype=torch.float32)
action_probabilities = torch.tensor(action_probabilities, dtype=torch.float32)
win_probabilities = torch.tensor(win_probabilities, dtype=torch.float32)

print(f"Shape of tensor game state: {game_states.size()}")
print(f"Shape of tensor action probabilities: {action_probabilities.size()}")
print(f"Shape of tensor win probabilities: {win_probabilities.size()}")


class GamesDataset(Dataset):

    def __init__(self, game_states, action_probabilities, win_probabilities):
        self.game_states = game_states
        self.action_probabilities = action_probabilities
        self.win_probabilities = win_probabilities

    def __len__(self):
        return len(self.game_states)
    
    def __getitem__(self, index):
        return {
            'input': self.game_states[index],
            'action_probabilities': self.action_probabilities[index].reshape(-1),
            'win_probabilities': self.win_probabilities[index].reshape(-1)
        }
    
dataset = GamesDataset(game_states, action_probabilities, win_probabilities)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

model = NeuralNet()
model.load_state_dict(torch.load(model_path))
print(device)
model.to(device)
model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def loss_function(target_action_p, sampled_action_p, game_result, sampled_value, c):

    value_loss = F.mse_loss(sampled_value, game_result)
    policy_loss = F.cross_entropy(sampled_action_p, target_action_p)
    l2_reg = torch.tensor(0.).to(sampled_value.device)
    for parameter in model.parameters():
        l2_reg += torch.norm(parameter)**2
    reg_loss = c * l2_reg

    return value_loss + policy_loss + reg_loss


losses = []
for epoch in range(4):
    total_loss = 0
    for batch in dataloader:
        inputs = batch['input'].to(device)
        target_action_probabilities = batch['action_probabilities'].to(device)
        target_win_probability = batch['win_probabilities'].to(device)

        sampled_action_probabilities, sampled_win_probability = model(inputs)

        loss = loss_function(target_action_probabilities, sampled_action_probabilities, target_win_probability, sampled_win_probability, c=0.0001)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    average_loss = total_loss / len(dataloader)
    losses.append(average_loss)

    print(f"Epoch {epoch+1}/{4}, Loss: {average_loss:.4f} ")

torch.save(model.state_dict(), model_path)
    

print("--- %s seconds ---" % (time.time() - start_time))
