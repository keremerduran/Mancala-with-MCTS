"""
Delete this
"""
import copy
import time
import os

import numpy as np 
import json

from Game import Game
from NN import NeuralNet
from MCTS import MCTS, MCTS_Node


start_time = time.time()

initial_board_state = MCTS_Node(np.array([4,4,4,4,4,4,0,4,4,4,4,4,4,0])) 
model = NeuralNet()
MCTS = MCTS()
Game = Game()


def play_games(state_node, model, tree_search, n_of_games):
    games = []
    for i in range(n_of_games):
        state = copy.deepcopy(state_node)
        print(f"Game {i}")
        player = "p1"
        while not state.game_over():

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

    file_path = r' '

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            data.extend(game_data)
    else:
        data = game_data

    with open(file_path, 'w') as file:
        json.dump(data, file)
    

if __name__ == '__main__':

    games = play_games(initial_board_state, model, MCTS, n_of_games=1000)
        
    save_games(games)

    print("--- %s seconds ---" % (time.time() - start_time))

