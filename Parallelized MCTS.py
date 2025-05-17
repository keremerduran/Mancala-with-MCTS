import torch
import torch.multiprocessing as mp
from multiprocessing import Barrier, Queue
import logging
import time
import numpy as np
from NN import NeuralNet  
from Game import Game

# Initialize the logger
logging.basicConfig(level=logging.INFO, format='%(processName)s - %(message)s')

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
        return [i for i in range(6) if self.state[i] > 0]
    
    def game_over(self):
        return self.state[6] > 24 or self.state[13] > 24 or sum(self.state[:6]) == 0 or sum(self.state[7:13]) == 0
    
    def is_leaf_node(self):
        return all(child is None for child in self.child_nodes)

class WorkerProcess(mp.Process):
    def __init__(self, root_node, process_id, barrier, queue, result_queue, termination_event, nn, max_simulations):
        super().__init__()
        self.process_id = process_id
        self.barrier = barrier
        self.queue = queue
        self.result_queue = result_queue
        self.termination_event = termination_event
        self.nn = nn
        self.max_simulations = max_simulations
        
        self.actions_history = []
        self.state_nodes_history = []
        self.game = Game()  # Initialize Game once per worker process
        self.root_node = root_node

    def run(self):
        try:
            simulations = 0
            total_search_time = 0.0
            total_infer_time = 0.0
            total_backprop_time = 0.0

            while simulations < self.max_simulations:
                start_time = time.time()
                state_node = self.search()
                total_search_time += time.time() - start_time

                if state_node is None or state_node.game_over():
                    self.queue.put((self.process_id, None))  # Signal terminal state
                else:
                    self.barrier.wait()
                    self.queue.put((self.process_id, state_node.state.tolist()))
                
                while True:
                    process_id, inference_result = self.result_queue.get()
                    if process_id == self.process_id:
                        break
                    self.result_queue.put((process_id, inference_result))  

                if state_node is not None and not state_node.game_over():
                    infer_start = time.time()
                    action_probabilities, value = inference_result
                    total_infer_time += time.time() - infer_start

                    expand_start = time.time()
                    self.expand(state_node, action_probabilities, value)
                    total_backprop_time += time.time() - expand_start

                    self.backpropagate(state_node)
                
                simulations += 1
            
            self.result_queue.put((self.process_id, self.root_node.N.tolist()))  

            logging.info(f"Process {self.process_id}: Total search time {total_search_time:.3f}s, inference {total_infer_time:.3f}s, backprop {total_backprop_time:.3f}s")
        
        except Exception as e:
            logging.error(f"Exception in process {self.process_id}: {e}")

    def search(self):
        current_state = self.root_node
        if current_state is None:
            return None
        
        self.state_nodes_history.append(current_state)
        while not current_state.is_leaf_node() and not current_state.game_over():
            action = self.action_selection(current_state)
            self.actions_history.append(action)
            if current_state.child_nodes[action] is None:
                action_board_index = current_state.possible_moves()[action]
                next_state = self.game.act(current_state.state, action_board_index)
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

    def expand(self, state_node, action_probabilities, value):
        state_node.P = action_probabilities.detach().numpy()[0]
        state_node.P = self.masking(state_node, state_node.P)
        state_node.v = value
        
        action = self.action_selection(state_node)
        action_board_index = state_node.possible_moves()[action]
        next_state = self.game.act(state_node.state, action_board_index)
        next_state = MCTS_Node(next_state)
        self.actions_history.append(action)
        state_node.child_nodes[action] = next_state
        return next_state

    def backpropagate(self, state_node):
        for i in range(len(self.state_nodes_history)):
            action = self.actions_history[-(i+1)]
            state_node = self.state_nodes_history[-(i+1)]
            state_node.N_b += 1
            state_node.N[action] += 1
            state_node.W[action] += state_node.v
            state_node.Q[action] = state_node.W[action] / state_node.N[action]

    def action_selection(self, state_node):
        epsilon = 0.25
        alpha = 0.03
        c_puct = 1.1
        dir = np.random.dirichlet([alpha] * state_node.n_of_children)   
        selection_values = state_node.Q + c_puct * ((1 - epsilon) * state_node.P + epsilon * dir)  * np.sqrt(state_node.N_b) / (1 + state_node.N)
        return np.argmax(selection_values)

    def masking(self, node, probs):
        legal_moves = node.possible_moves()
        masked = np.zeros_like(probs)
        for i, move in enumerate(legal_moves):
            masked[i] = probs[move]
        masked_sum = np.sum(masked)
        if masked_sum > 0:
            masked /= masked_sum
        return masked

def main():
    num_processes = 4
    max_simulations = 25
    model_path = "pretrained_model.pth"
    initial_state = MCTS_Node(np.array([4,4,4,4,4,4,0,4,4,4,4,4,4,0])) 

    model = NeuralNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    queue = Queue()
    result_queue = Queue()
    termination_event = mp.Event()
    barrier = Barrier(num_processes)
    
    processes = [WorkerProcess(initial_state, i, barrier, queue, result_queue, termination_event, model, max_simulations) for i in range(num_processes)]
    for p in processes:
        p.start()
    
    active_processes = set(range(num_processes))
    try:
        while active_processes:
            batch = []
            process_ids = []
            
            while len(batch) < len(active_processes):
                process_id, game_state = queue.get()
                if game_state is None:
                    active_processes.remove(process_id)
                else:
                    batch.append(torch.tensor(game_state).float().unsqueeze(0))
                    process_ids.append(process_id)
            
            if batch:
                input_batch = torch.cat(batch, dim=0)
                with torch.no_grad():
                    outputs = model(input_batch)
                for i, process_id in enumerate(process_ids):
                    result_queue.put((process_id, outputs[i]))
                  
        visit_counts = np.zeros(len(initial_state.moves))
        for _ in range(num_processes):
            process_id, N_values = result_queue.get()
            visit_counts += np.array(N_values)
    
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Terminating processes.")
    
    finally:
        termination_event.set()
        for p in processes:
            p.join()
          
    logging.info(f"Aggregated visit counts: {visit_counts}")
    return visit_counts

if __name__ == "__main__":
    main()