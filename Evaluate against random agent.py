import numpy as np 
from Game import Game
from NN import NeuralNet
from MCTS import MCTS, MCTS_Node
import copy
import time
import torch

start_time = time.time()

model_path = r'C:\Users\Bogazici\Desktop\MCTS with NNs\models\10_05_25_MANGO_sixthraining8000games_64batchsize.pth'

initial_board_state = MCTS_Node(np.array([4,4,4,4,4,4,0,4,4,4,4,4,4,0])) 

MCTS = MCTS()
model = NeuralNet()
model.load_state_dict(torch.load(model_path))
model.eval()
Game = Game()

def play_games(state_node, model, tree_search, game, n_of_games):
    games = []
    for i in range(n_of_games):
        state = copy.deepcopy(state_node)
        print(f"Game {i}")
        player = np.random.choice(["p1", "p2"])
        while not state.game_over():

            # The NN with MCTS is player 1
            if player == "p1":
                action, _, altered_state_node = tree_search.simulate(state_node=state, model=model)

                next_state = altered_state_node.child_nodes[action]
                action = state.possible_moves()[action]

            # Player 2 is a random agent
            else:
                action = np.random.choice(range(state.n_of_children))
                action = state.moves[action] # This line is necessary so that the index of the chosen action matches the correct index on the board
                next_state = game.act(state.state, action)
                next_state = MCTS_Node(next_state)
                        
            # These lines track which player was taking action during that state so that the win/loss information can be appended after the game ends
            if (state.state[action] + action) % 13 != 6:
                player = "p2" if player == "p1" else "p1"

            state = next_state

        # Change the "p1" and "p2" to 0 or 1 where 0 represents the winner and 1 represents the loser
        winner = Game.determine_winner(state.state.tolist(), player)
        games.append(1) if winner == "p1" else games.append(0) 
           
    return games

games = play_games(initial_board_state, model, MCTS, Game, n_of_games=1000)

print(f"Winrate of agent against random opponent: {round(sum(games)/len(games), 3)}")

print("--- %s seconds ---" % (time.time() - start_time))

