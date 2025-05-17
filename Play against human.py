"""
delete this 

"""

import copy
import time

import numpy as np
import torch

from Game import Game
from NN import NeuralNet
from MCTS import MCTS, MCTS_Node


start_time = time.time()


def play(state_node, model, tree_search, game):

    state = copy.deepcopy(state_node)
    player = np.random.choice(["p1", "p2"])
    print(f"Current state: {state.state}")
    while not state.game_over():
        # The NN with MCTS is player 1
        if player == "p1":
            action, _, altered_state_node = tree_search.simulate(state_node=state, model=model)
            next_state = altered_state_node.child_nodes[action]
            action = state.possible_moves()[action]
            print(f"Selected action during actual game: {action + 1}")
            print(f"Next state has become:  {next_state.state}")

        # Player 2 is us
        else:
            print("Please select an action: (1-6)")
            while True:
                action = input()
                action = int(action) - 1
                if action < 0 or action > 5 or type(action) != int:
                    print("Invalid action. Please select a number between 1 and 6.")
                    continue
                elif state.state[action] == 0:
                    print("Invalid action. Please select a valid move.")
                    continue
                break
            next_state = game.act(state.state, action)
            next_state = MCTS_Node(next_state)     
            flipped_board_state = np.concatenate((next_state.state[7:14], next_state.state[0:7]))           
            print(f"Next state has become:  {flipped_board_state}")

        # These lines track which player was taking action during that state so that the win/loss information can be appended after the game ends
        if (state.state[action] + action) % 13 != 6:
            player = "p2" if player == "p1" else "p1"

        state = next_state
    
    winner = Game.determine_winner(state.state.tolist())
    
    if winner == "p1":
        print("You lose!")
    elif winner == "p2":
        print("You win!")   
    elif winner == None:
        print("It's a draw!")
    
    print(f"Game finished in {time.time() - start_time} seconds.")


if __name__ == "__main__":

    MCTS = MCTS()
    Game = Game()

    model_path = ' '
    initial_board_state = MCTS_Node(np.array([4,4,4,4,4,4,0,4,4,4,4,4,4,0]))
    model = NeuralNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    play(initial_board_state, model, MCTS, Game)

