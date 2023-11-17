# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.depth_limit = 3  # Depth limit for the search


    def step(self, chess_board, my_pos, adv_pos, max_step):
        start_time = time.time()

        # Implement Alpha-Beta Pruning within this function
        best_move, _ = self.alpha_beta(chess_board, my_pos, adv_pos, max_step, self.depth_limit, -np.inf, np.inf, True)

        time_taken = time.time() - start_time
        print("My AI's turn took ", time_taken, "seconds.")

        return best_move

    def alpha_beta(self, board, my_pos, adv_pos, max_step, depth, alpha, beta, maximizing_player):

        if depth == 0 or self.is_terminal(board, my_pos, adv_pos):
            return None, self.evaluate(board, my_pos, adv_pos)

        if maximizing_player:
            max_eval = -np.inf
            best_move = None
            for move in self.get_possible_moves(board, my_pos, adv_pos, max_step):
                eval = self.alpha_beta(self.simulate_move(board, move), move[0], adv_pos, max_step, depth - 1, alpha, beta, False)[1]
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return best_move, max_eval
        else:
            min_eval = np.inf
            best_move = None
            for move in self.get_possible_moves(board, adv_pos, my_pos, max_step):
                eval = self.alpha_beta(self.simulate_move(board, move), my_pos, move[0], max_step, depth - 1, alpha, beta, True)[1]
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return best_move, min_eval
    
    def get_possible_moves(self, board, pos, adv_pos, max_step):
        moves_list = []
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
        rows, cols = len(board), len(board[0])

        # Function to check if a position is within the board and not blocked by a barrier
        def is_valid_move(row, col, dir_idx):
            return 0 <= row < rows and 0 <= col < cols and not board[row][col][dir_idx]

        # Explore each direction
        for dir_idx, (dr, dc) in enumerate(directions):
            for step in range(1, max_step + 1):
                new_row, new_col = pos[0] + dr * step, pos[1] + dc * step

                # Check if the new position is valid
                if is_valid_move(pos[0] + dr * (step - 1), pos[1] + dc * (step - 1), dir_idx):
                    # Add the move with each possible barrier placement
                    for barrier_dir in self.dir_map.values():
                        moves_list.append(((new_row, new_col), barrier_dir))
                else:
                    # If the path is blocked or out of bounds, stop exploring further in this direction
                    break

        return moves_list



    def simulate_move(self, board, move):
        new_board = deepcopy(board)
        pos, dir = move
        # Place the barrier in the chosen direction
        new_board[pos[0]][pos[1]][dir] = True
        return new_board
    
    def evaluate(self, board, my_pos, adv_pos):
        my_score = self.calculate_zone_score(board, my_pos)
        adv_score = self.calculate_zone_score(board, adv_pos)
        return my_score - adv_score

    def calculate_zone_score(self, board, pos):
        """
        Calculate the score for a zone. This function should consider not only the current
        number of free spaces but also the potential for future expansion and control.
        """
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left

        score = 0
        visited = set()
        queue = [pos]

        while queue:
            current_pos = queue.pop(0)
            if current_pos in visited:
                continue

            visited.add(current_pos)
            r, c = current_pos

            # Check all four directions from the current position
            for d in range(4):  # Up, Right, Down, Left
                if not board[r][c][d]:
                    dr, dc = directions[d]
                    next_pos = (r + dr, c + dc)
                    if 0 <= next_pos[0] < len(board) and 0 <= next_pos[1] < len(board[0]):
                        queue.append(next_pos)
                        score += 1
                        # Additional scoring can be added here based on strategic considerations,
                        # such as proximity to the opponent, control of the central area, etc.

        return score

    def is_terminal(self, board, my_pos, adv_pos):
        # Check if any player is completely enclosed
        return not any(~board[my_pos[0]][my_pos[1]]) or not any(~board[adv_pos[0]][adv_pos[1]])
    


