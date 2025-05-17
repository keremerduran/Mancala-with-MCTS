"""
delete this

"""
import numpy as np 
from Game import Game
from NN import NeuralNet
from MCTS import MCTS, MCTS_Node
import copy
import time
import json
import os
import torch


start_time = time.time()

model_path = r'C:\Users\Bogazici\Desktop\MCTS with NNs\Data\Models\model_trained_on_9k_randoms.pth'

initial_board_state = MCTS_Node(np.array([4,4,4,4,4,4,0,4,4,4,4,4,4,0])) 

model = NeuralNet()
model.load_state_dict(torch.load(model_path))
model.eval()
Game = Game()
MCTS = MCTS()


def play_games(state_node, model, n_of_games):
    games = []
    for i in range(n_of_games):
        state = copy.deepcopy(state_node)
        print(f"Game {i}")
        player = "p1"
        while not state.game_over():

            board_state = torch.tensor(state.state.tolist(), dtype=torch.float)
            board_state = board_state.unsqueeze(0)
            action_probabilities, _ = model(board_state)
            action_probabilities = action_probabilities.detach().numpy()[0]
            print(action_probabilities)
            masked_action_probabilities = MCTS.masking(state, action_probabilities)
            print(state.state)
            print(masked_action_probabilities)


            action = np.searchsorted(np.cumsum(masked_action_probabilities), np.random.sample())
            print(action)
            action = state.moves[action] # This line is necessary so that the index of the chosen action matches the correct index on the board
            next_state = Game.act(state.state, action)
            next_state = MCTS_Node(next_state)

            masked_action_probabilities = []
            count = 0
            for i in range(6):
                if state.state[i] == 0:
                    masked_action_probabilities.append(0.0)
                else:
                    action_probabilities[count] = round(action_probabilities[count], 3)
                    masked_action_probabilities.append(action_probabilities[count])
                    count += 1

            games.append([state.state.tolist(), masked_action_probabilities, player])

            # These lines track which player was taking action during that state so that the win/loss information can be appended after the game ends
            if (state.state[action] + action) % 13 != 6:
                player = "p2" if player == "p1" else "p1"

            state = next_state
        
        # Change the "p1" and "p2" to 0 or 1 where 0 represents the winner and 1 represents the loser
        winner = Game.determine_winner(state.state)
        for state in games:
            state[-1] = 0 if state[-1] == winner else 1
                    
    return games


def save_games(game_data):

    file_path = r'C:\Users\Bogazici\Desktop\MCTS with NNs\Data\Games\semi_random_games.json'

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            data.extend(game_data)
    else:
        data = game_data

    with open(file_path, 'w') as file:
        json.dump(data, file)
    

games = play_games(initial_board_state, model, n_of_games=10)

#save_games(games)


print("--- %s seconds ---" % (time.time() - start_time))
