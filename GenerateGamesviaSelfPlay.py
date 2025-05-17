"""
This is the main game generation script. It uses a pretrained model to generate a specified number of 
games where each action is selected using a combination of that model and Monte Carlo Tree Search.

"""

import copy
import time
import os

import json
import numpy as np 
import torch

from Game import Game
from NN import NeuralNet
from MCTS import MCTS, MCTS_Node


start_time = time.time()


model_path = ' '

initial_board_state = MCTS_Node(np.array([4,4,4,4,4,4,0,4,4,4,4,4,4,0])) 

model = NeuralNet()
model.load_state_dict(torch.load(model_path))
model.eval()

MCTS = MCTS()
Game = Game()


def play_games(state_node, model, tree_search, n_of_games):

    games = []
    for i in range(n_of_games):
        state = copy.deepcopy(state_node)  
        if i % 100 == 0:
            print(f"Game {i}")
        player = "p1"
        game_length = 0
        while not state.game_over():
            game_length += 1
            action, action_probabilities, altered_state_node = tree_search.simulate(state_node=state, model=model)
            next_state = altered_state_node.child_nodes[action]

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
            action_board_index = state.possible_moves()[action]
            if (state.state[action_board_index] + action_board_index) % 13 != 6:
                player = "p2" if player == "p1" else "p1"

            state = next_state
        
        winner = Game.determine_winner(state.state.tolist(), player)

        for state in games[-game_length:]:
            state[-1] = 1 if state[-1] == winner else 0 if state[-1] == None else -1


    return games


def save_games(game_data):

    file_path = ' '

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            data.extend(game_data)
    else:
        data = game_data

    with open(file_path, 'w') as file:
        json.dump(data, file)
    

if __name__ == '__main__':

    games = play_games(initial_board_state, model, MCTS, n_of_games=8000)

    save_games(games)

    print("--- %s seconds ---" % (time.time() - start_time))

