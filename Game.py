import numpy as np
import copy

class Game():
    """
    def __init__(self, board_state=None):
        self.board_state = board_state if board_state is not None else np.array([4,4,4,4,4,4,0,4,4,4,4,4,4,0])
    """
    def act(self, current_board_state, action):
        board_state = copy.deepcopy(current_board_state)
        if action > 5:
            raise ValueError("This isn't an allowed action")
        # We remove the last element of the board state (we will eventually add it back at the end) which corresponds to the enemy treasure because no action we take can affect it and it is more convenient this way
        enemy_treasure = board_state[-1]
        board_state = board_state[:-1]
        value = board_state[action]
        end_index = (value + action) % 13

        if value != 1:
            self.distribute_stones(board_state, action, value - 1)
        else:
            board_state[action] = 0

        # If the last stone ends up in one of our empty wells we take all the stones inside the opposite well
        if end_index < 6 and board_state[end_index] == 0 and board_state[12 - end_index] != 0:
            board_state[6] += board_state[12 - end_index] + 1
            board_state[12 - end_index] = 0
            board_state = np.append(board_state, enemy_treasure)
            board_state = np.concatenate((board_state[7:14], board_state[0:7]))

        # If the last stone ends up in our treasure we play again which is why the board isn't flipped
        elif end_index == 6:
            board_state[6] += 1
            board_state = np.append(board_state, enemy_treasure)

        else:
            board_state[end_index] += 1
            board_state = np.append(board_state, enemy_treasure)
            board_state = np.concatenate((board_state[7:14], board_state[0:7]))

        return board_state
    
    def distribute_stones(self, board_state, action, value):
        if value == 0:
            raise ValueError('There are no stones inside that well')
        board_state[action] = 0
        for i in range(value):
            board_state[(action + i + 1) % 13] += 1

    def determine_winner(self, board_state):
        if sum(board_state[:6]) == 0:
            board_state[6] += sum(board_state[7:13])
        elif sum(board_state[7:13]) == 0:
            board_state[13] += sum(board_state[:6])
        if board_state[6] > 24: 
            return "first_player"
        elif board_state[13] > 24:
            return "second_player"
        elif board_state[6] == 24 and board_state[6] == 24:
            return "draw"

