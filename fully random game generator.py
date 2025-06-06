"""
This script generates games where actions on both sides are selected randomly
This is useful because random games are significantly faster to generate and even though
game quality is very low, they can still serve to pretrain the model.

"""

import os
import copy
import time

import numpy as np 
import json

from Game import Game
from MCTS import MCTS_Node

start_time = time.time()

initial_board_state = MCTS_Node(np.array([4,4,4,4,4,4,0,4,4,4,4,4,4,0])) 

Game = Game()


def play_games(state_node, game, n_of_games):

    games = []
    for i in range(n_of_games):
        state = copy.deepcopy(state_node)
        player = "p1"
        game_length = 0
        while not state.game_over():
            game_length += 1

            random_probabilities = np.random.rand(state.n_of_children) 
            r_action_probabilities = random_probabilities / np.sum(random_probabilities) # Normalizing probabilities
            action = np.random.choice(range(state.n_of_children), p=r_action_probabilities) # Randomly choosing an action with respect to the probabilities
            action = state.moves[action] # This line is necessary so that the index of the chosen action matches the correct index on the board
            next_state = game.act(state.state, action) # Taking that action and receiving the next state as an array
            next_state = MCTS_Node(next_state) # Converting the array into a MCTS_Node object 

            masked_action_probabilities = []
            count = 0
            for i in range(6):
                if state.state[i] == 0:
                    masked_action_probabilities.append(0.0)
                else:
                    r_action_probabilities[count] = np.round(r_action_probabilities[count], 3)
                    masked_action_probabilities.append(r_action_probabilities[count])
                    count += 1

            games.append([state.state.tolist(), masked_action_probabilities, player])

            # These lines track which player was taking action during that state so that the win/loss information can be appended after the game ends
            if (state.state[action] + action) % 13 != 6:
                player = "p2" if player == "p1" else "p1"

            state = next_state

        winner = Game.determine_winner(state.state.tolist(), player)

        for state in games[-game_length:]:
            state[-1] = 1 if state[-1] == winner else 0 if state[-1] == None else -1

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
    games = play_games(initial_board_state, Game, n_of_games=4000)

    save_games(games)

    print("--- %s seconds ---" % (time.time() - start_time))
