import numpy as np
from Game import Game
from NN import NeuralNet
import torch
import time
#import multiprocessing as mp
import copy

start_time = time.time()


class MCTS_Node:
    def __init__(self, state):
        self.state = state
        self.moves = self.possible_moves()
        self.n_of_children = len(self.possible_moves())
        self.N_b = 1 # Maybe this should be initialised as 0 I'm not sure yet
        self.N = np.zeros(self.n_of_children) # Visit count
        self.W = np.zeros(self.n_of_children) # Total action value
        self.Q = np.zeros(self.n_of_children) # Mean action value
        self.P = np.zeros(self.n_of_children) # Prior probability  -- this should be initialized by querying the NN --
        self.child_nodes = np.empty(self.n_of_children, dtype=object)
        self.child_nodes[:] = None
        self.v = 0
        #self.lock = mp.Lock()

    def possible_moves(self):
        moves = [i for i in range(6) if self.state[i] > 0]
        return moves
    
    def game_over(self):
        # Is the correct place for this method inside this class?
        if self.state[6] > 24 or self.state[13] > 24:
            return True
        elif sum(self.state[:6]) == 0 or sum(self.state[7:13]) == 0:
            return True
        else:
            return False
    
    def is_leaf_node(self):
        # If necessary this method can be deleted and replaced with direct checks where needed for efficieny but at the cost of readability
        empty_child_nodes = np.empty(self.n_of_children, dtype=object)
        empty_child_nodes[:] = None 
        return np.array_equal(self.child_nodes, empty_child_nodes)
    
    '''
    def __deepcopy__(self, memo):
        # This was necessary in order to copy locks, since I no longer use them, this can be discarded
        copied_node = MCTS_Node(copy.deepcopy(self.state, memo))

        copied_node.state = self.state
        copied_node.moves = self.moves
        copied_node.n_of_children = self.n_of_children
        copied_node.N_b = self.N_b 
        copied_node.N = self.N
        copied_node.W = self.W
        copied_node.Q = self.Q
        copied_node.P = self.P
        copied_node.child_nodes = copy.deepcopy(self.child_nodes, memo)
        copied_node.v = self.v
        copied_node.lock = mp.Lock()

        return copied_node
    '''
    

class MCTS:

    def __init__(self):
        self.actions_history = []
        self.state_nodes_history = []
        self.expansion_time = 0
        self.search_time = 0
        self.backprop_time = 0
        
    def search(self, state_node):
        #print("--- SEARCH ---")
        current_state = state_node
        self.state_nodes_history.append(current_state)

        while not current_state.is_leaf_node() and not current_state.game_over():
            #print(f"Currently traversing through {current_state.state} while searching")
            action = self.action_selection(current_state)
            self.actions_history.append(action)
            if current_state.child_nodes[action] == None:
                game = Game()
                action_board_index = current_state.possible_moves()[action]
                #print(f"Selected action during search (indexed wrt. the board): {action_board_index}")

                next_state = game.act(current_state.state, action_board_index) 
                next_state = MCTS_Node(next_state)
                if not next_state.game_over():
                    self.state_nodes_history.append(next_state)

                current_state.child_nodes[action] = next_state
            else:   
                #print(f"Selected action during search (indexed wrt. possible actions): {action}")

                next_state = current_state.child_nodes[action]
                if not next_state.game_over():
                    self.state_nodes_history.append(next_state)

            current_state = next_state
            #print(f"Next state has become {current_state.state}")

        return current_state   
            
    def expand(self, state_node, model):
        # The way expansion is implemented here is once you reach a leaf node, the NN is queried and based on the returned action probabilities an action
        # is selected and then one of the actions is expanded so the state now has a non-empty child. Maybe the way expansion should be implemented is 
        # only the NN is queried and no action selection takes place. I should go back to the original paper and check this.
        #print("--- EXPANSION ---")
        #print(f"Expanding the leaf node that is: {state_node.state}")
        board_state = torch.tensor(state_node.state.tolist(), dtype=torch.float)
        board_state = board_state.unsqueeze(0)

        # Unfortunately the overhead of transfering the tensor and the model to the GPU and loading it back takes more time then simply doing the calculation on the CPU.
        # This might become viable if I ever do parallel simulations

        #board_state = board_state.unsqueeze(0).to('cuda')
        #model = model.to('cuda')

        action_probabilities, value = model(board_state)

        #action_probabilities = action_probabilities.cpu() you don't need this line just add .cpu() between detach and numpy below

        state_node.P = action_probabilities.detach().numpy()[0]
        state_node.P = self.masking(state_node, state_node.P)
        state_node.v = value 
        #print(f"Action probabilities for the state are: {state_node.P}")
        #print(f"Current state value is: {state_node.v}")
        action = self.action_selection(state_node)
        action_board_index = state_node.possible_moves()[action]
        #print(f"Selected action during expansion: {action_board_index}")
        game = Game()
        next_state = game.act(state_node.state, action_board_index)
        next_state = MCTS_Node(next_state)
        #print(f"Next state has become: {next_state.state}")
        self.actions_history.append(action)
        state_node.child_nodes[action] = next_state


    def backpropogate(self):
        #print("--- BACKPROP ---")
        #print(f"Action history length for backpropogation: {len(self.actions_history)}")
        #print(f"State history length for backpropogation: {len(self.state_nodes_history)}")
        for i in range(len(self.state_nodes_history)):
            action = self.actions_history[-(i+1)]
            state_node = self.state_nodes_history[-(i+1)]
            #print(f"Current state during backprop: {state_node.state}")
            #print(f"Current action during backprop: {action}")
            state_node.N_b += 1
            state_node.N[action] += 1
            state_node.W[action] += state_node.v
            state_node.Q[action] = state_node.W[action] / state_node.N[action]

    def masking(self, state_node, probs, expansion=True):
        #print("--- MASKING ---")
        #print(f"Probabilities before masking: {probs}")
        masked_probs = []
        if expansion:
            # This ensures that we only get the probabilities of legal moves (i.e. moves corresponding to non-empty wells)
            for i in range(len(state_node.moves)):
                masked_probs.append(probs[state_node.moves[i]])
        else:
            masked_probs = probs
        clamped_probs = list(map(lambda x: max(x,0), masked_probs)) #This might be unnecessary now since I switched to softmax activation
        prob_sum  = sum(masked_probs)
        normalized_probs = [p / prob_sum if prob_sum > 0 else 0 for p in clamped_probs]
        #print(f"Probabilities after masking: {np.array(normalized_probs)}")
        return np.array(normalized_probs)


    def action_selection(self, state_node):
        #print("--- ACTION SELECT ---")
        epsilon = 0.25
        alpha = 0.03      # This hyperparameter needs tuning
        c_puct = 1.1      # I don't actually know what the appropriate value for this constant is either

        dir = np.random.dirichlet([alpha] * state_node.n_of_children)   
        #print(f"Adding dirichlet noise: {dir}")
        selection_values = state_node.Q + c_puct * ((1 - epsilon) * state_node.P + epsilon * dir)  * np.sqrt(state_node.N_b) / (1 + state_node.N)
        #print(f"Selection values during action selection (indexed wrt. possible actions): {selection_values}")
        return np.argmax(selection_values)
    

    def simulate(self, state_node, model):
        #print("--- SIMULATION ---")
        n_of_simulations = 40
        for i in range(n_of_simulations):
            #print(f"SIMULATION {i}")
            #search_start_time = time.time() 
            leaf_node = self.search(state_node)
            #search_end_time = time.time()
            #self.search_time += search_end_time - search_start_time
            temp = 1
            move = 0
            if not leaf_node.game_over():
                if move > 6: 
                    temp = 0.00000001
                #expansion_start_time = time.time() 
                self.expand(leaf_node, model)
                move += 1
                #expansion_end_time = time.time()
                #self.expansion_time += expansion_end_time - expansion_start_time
            if i != 0: # Pretty sure this should be move count instead of game count
                #backprop_start_time = time.time() 
                self.backpropogate()
                #backprop_end_time = time.time()
                #self.backprop_time += backprop_end_time - backprop_start_time
                state_node.N[self.actions_history[0]] += 1  
            self.actions_history = []
            self.state_nodes_history = []
        state_node.N_b += n_of_simulations # This is also updated during backprop so is it being updated twice?

        action_probabilities = [(state_node.N[action] / state_node.N_b)**(1/temp)  for action in range(state_node.n_of_children)]
        masked_action_probabilities = self.masking(state_node, action_probabilities, expansion=False)  

        #print(f"Search time {self.search_time}")
        #print(f"Expansion time {self.expansion_time}")
        #print(f"Backprop time {self.backprop_time}")

        return [np.searchsorted(np.cumsum(masked_action_probabilities), np.random.sample()), action_probabilities, state_node] # Changed the return to include the state node so that the tree can be preserved
    

"""
model = NeuralNet()
search_tree = MCTS()    
initial_board_position = MCTS_Node(np.array([4,4,4,4,4,4,0,4,4,4,4,4,4,0])) 

action, probabilities = search_tree.simulate(initial_board_position, model)
print(f"Selected action is {action}")
"""

