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

file_path = r'C:\Users\erdur\Desktop\Using Neural Networks and MCTS to play Mancala\games\fully_random_games2.json'
model_path = r'C:\Users\erdur\Desktop\Using Neural Networks and MCTS to play Mancala\models\model_trained_on_random_games.pth'

initial_board_state = MCTS_Node(np.array([4,4,4,4,4,4,0,4,4,4,4,4,4,0])) 
model = NeuralNet()
#model.load_state_dict(torch.load(model_path))
#model.eval()
MCTS = MCTS()
Game = Game()


def play_games(state_node, model, tree_search, n_of_games):
    games = []
    for i in range(n_of_games):
        state = copy.deepcopy(state_node)
        print(f"Game {i}")
        player = "p1"
        while not state.game_over():
            print(f"PLAYER: {player}")
            print(f"State: {state.state}")
            action, action_probabilities, altered_state_node = tree_search.simulate(state_node=state, model=model)
            print(f"Selected action during game: {action}")
            next_state = altered_state_node.child_nodes[action]
            print(f"Next state: {next_state}")

            masked_action_probabilities = []
            count = 0
            for i in range(6):
                if state.state[i] == 0:
                    masked_action_probabilities.append(0.0)
                else:
                    action_probabilities[count] = round(action_probabilities[count], 3)
                    masked_action_probabilities.append(action_probabilities[count])
                    count += 1

            games.append([state.state.tolist(), round(masked_action_probabilities, 3), player])

            # These lines track which player was taking action during that state so that the win/loss information can be appended after the game ends
            action_board_index = state.possible_moves()[action]
            if (state.state[action_board_index] + action_board_index) % 13 != 6:
                player = "p2" if player == "p1" else "p1"

            state = next_state
        
        games.append([state.state.tolist()])
        
        # Change the "p1" and "p2" to 0 or 1 where 0 represents the winner and 1 represents the loser
        winner = Game.determine_winner(games[-1][0])
        if winner == "first_player":
            if player == "p1":
                winner = "p1"
            else:
                winner = "p2"
        elif winner == "second_player":
            if player == "p1":
                winner = "p2"
            else:
                winner = "p1"
        else:
            winner = None

        games.pop()

        for state in games:
            if state[-1] == winner:
                state[-1] = 1
            elif winner == None:
                state[-1] = 0
            else:
                state[-1] = -1
            
            # state[-1] = 0 if state[-1] == winner else 1
        

    return games


def save_games(game_data):

    file_path = r'C:\Users\erdur\Desktop\Using Neural Networks and MCTS to play Mancala\games\2_debugging_games.json'

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            data.extend(game_data)
    else:
        data = game_data

    with open(file_path, 'w') as file:
        json.dump(data, file)
    

games = play_games(initial_board_state, model, MCTS, n_of_games=2)

save_games(games)


print("--- %s seconds ---" % (time.time() - start_time))
