"""
This script uses two pretrained models to play games and compare performance over a specified number of games.

"""

import copy
import time

import torch
import numpy as np 
import json

from Game import Game
from NN import NeuralNet
from MCTS import MCTS, MCTS_Node


start_time = time.time()


def play_games(state_node, first_model, second_model, tree_search, n_of_games):
    games = []
    for i in range(n_of_games):
        print(f"Self play evaluation game {i+1}/{n_of_games}")
        state = copy.deepcopy(state_node)
        player = np.random.choice(["p1", "p2"])
        while not state.game_over():
            model = first_model if player == "p1" else second_model
            action, _, altered_state_node = tree_search.simulate(state_node=state, model=model)

            next_state = altered_state_node.child_nodes[action]
            action = state.possible_moves()[action]
                        
            # These lines track which player was taking action during that state so that the win/loss information can be appended after the game ends
            if (state.state[action] + action) % 13 != 6:
                player = "p2" if player == "p1" else "p1"

            state = next_state

        # Change the "p1" and "p2" to 0 or 1 where 0 represents the winner and 1 represents the loser
        winner = Game.determine_winner(state.state.tolist(), player)
        games.append(1) if winner == "p1" else games.append(0) if winner == None else games.append(-1)

                    
    return games


if __name__ == "__main__":

    model1_path = r''
    model2_path = r''
    initial_board_state = MCTS_Node(np.array([4,4,4,4,4,4,0,4,4,4,4,4,4,0])) 

    Game = Game()
    MCTS = MCTS()

    model1 = NeuralNet()
    model1.load_state_dict(torch.load(model1_path))
    model1.eval()

    model2 = NeuralNet()
    model2.load_state_dict(torch.load(model2_path))
    model2.eval()

    n_of_games = 1000
    games = play_games(initial_board_state, model1, model2, MCTS, n_of_games)

    n_of_model1_wins = games.count(1)
    n_of_model2_wins = games.count(-1)
    n_of_draws = games.count(0)
    print(f"Model 1 points: {n_of_model1_wins + n_of_draws * 0.5}")
    print(f"Model 2 points: {n_of_model2_wins + n_of_draws * 0.5}")

    # Save the games 
    save_path = " "
    with open(save_path, 'w') as file:
        json.dump(games, file)

    print("--- %s seconds ---" % (time.time() - start_time))

