"""
This file contains the Monte Carlo Tree Search logic. It takes a game state and a pretrained model and carries out a specified number of simulations 
- where each simulation consists of search expand and backpropogate steps - and at the end returns visit counts of all the next states.
"""


import time
import copy

import numpy as np
import torch

from Game import Game
from NN import NeuralNet


start_time = time.time()


class MCTS_Node:

    def __init__(self, state):
        self.state = state
        self.moves = self.possible_moves()
        self.n_of_children = len(self.possible_moves())
        self.N_b = 1 
        self.N = np.zeros(self.n_of_children) # Visit count
        self.W = np.zeros(self.n_of_children) # Total action value
        self.Q = np.zeros(self.n_of_children) # Mean action value
        self.P = np.zeros(self.n_of_children) # Prior probability  
        self.child_nodes = np.empty(self.n_of_children, dtype=object) 
        self.child_nodes[:] = None
        self.v = 0

    def possible_moves(self):
        moves = [i for i in range(6) if self.state[i] > 0]
        return moves
    
    def game_over(self):
        if self.state[6] > 24 or self.state[13] > 24:
            return True
        elif sum(self.state[:6]) == 0 or sum(self.state[7:13]) == 0:
            return True
        else:
            return False
    
    def is_leaf_node(self):
        empty_child_nodes = np.empty(self.n_of_children, dtype=object)
        empty_child_nodes[:] = None 
        return np.array_equal(self.child_nodes, empty_child_nodes)
    

class MCTS:

    def __init__(self):
        self.actions_history = []
        self.state_nodes_history = []
        self.expansion_time = 0
        self.search_time = 0
        self.backprop_time = 0
        
    def search(self, state_node):

        current_state = state_node
        self.state_nodes_history.append(current_state)

        while not current_state.is_leaf_node() and not current_state.game_over():
            action = self.action_selection(current_state)
            self.actions_history.append(action)
            if current_state.child_nodes[action] == None:
                game = Game()
                action_board_index = current_state.possible_moves()[action]

                next_state = game.act(current_state.state, action_board_index) 
                next_state = MCTS_Node(next_state)
                if not next_state.game_over():
                    self.state_nodes_history.append(next_state)

                current_state.child_nodes[action] = next_state
            else:   
                next_state = current_state.child_nodes[action]
                if not next_state.game_over():
                    self.state_nodes_history.append(next_state)

            current_state = next_state

        return current_state   
            
    def expand(self, state_node, model):

        board_state = torch.tensor(state_node.state.tolist(), dtype=torch.float)
        board_state = board_state.unsqueeze(0)

        # Unfortunately the overhead of transfering the tensor and the model to the GPU and loading it back takes more time then simply doing the calculation on the CPU.
        # This might be for parallel simulations 

        action_probabilities, value = model(board_state)

        state_node.P = action_probabilities.detach().numpy()[0]
        state_node.P = self.masking(state_node, state_node.P)
        state_node.v = value 

        action = self.action_selection(state_node)
        action_board_index = state_node.possible_moves()[action]

        game = Game()
        next_state = game.act(state_node.state, action_board_index)
        next_state = MCTS_Node(next_state)

        self.actions_history.append(action)
        state_node.child_nodes[action] = next_state


    def backpropogate(self):

        for i in range(len(self.state_nodes_history)):
            action = self.actions_history[-(i+1)]
            state_node = self.state_nodes_history[-(i+1)]

            state_node.N_b += 1
            state_node.N[action] += 1
            state_node.W[action] += state_node.v
            state_node.Q[action] = state_node.W[action] / state_node.N[action]

    def masking(self, state_node, probs, expansion=True):

        masked_probs = []
        if expansion:
            # This ensures that we only get the probabilities of legal moves (i.e. moves corresponding to non-empty wells)
            for i in range(len(state_node.moves)):
                masked_probs.append(probs[state_node.moves[i]])
        else:
            masked_probs = probs
        clamped_probs = list(map(lambda x: max(x,0), masked_probs)) 
        prob_sum  = sum(masked_probs)
        normalized_probs = [p / prob_sum if prob_sum > 0 else 0 for p in clamped_probs]

        return np.array(normalized_probs)


    def action_selection(self, state_node):
        epsilon = 0.25
        alpha = 0.03      

        dir = np.random.dirichlet([alpha] * state_node.n_of_children)   
        selection_values = state_node.Q + c_puct * ((1 - epsilon) * state_node.P + epsilon * dir)  * np.sqrt(state_node.N_b) / (1 + state_node.N)

        return np.argmax(selection_values)
    

    def simulate(self, state_node, model):
        n_of_simulations = 40
        for i in range(n_of_simulations):
            leaf_node = self.search(state_node)
            temp = 1
            move = 0
            if not leaf_node.game_over():
                if move > 6: 
                    temp = 0.00000001
                self.expand(leaf_node, model)
                move += 1
            if i != 0: 
                self.backpropogate()
                state_node.N[self.actions_history[0]] += 1  
            self.actions_history = []
            self.state_nodes_history = []
        state_node.N_b += n_of_simulations 

        action_probabilities = [(state_node.N[action] / state_node.N_b)**(1/temp)  for action in range(state_node.n_of_children)]
        masked_action_probabilities = self.masking(state_node, action_probabilities, expansion=False)  

        return [np.searchsorted(np.cumsum(masked_action_probabilities), np.random.sample()), action_probabilities, state_node] 
    
