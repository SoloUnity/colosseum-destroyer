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

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, self.dir_map["u"]
    
    def minimax(self, myPos, advPos, depth, isMaximizing, maxStep):
        if depth == 0 or self.isGG():
            return self.eval()
        
        if isMaximizing:
            maxEval = float('-inf')
            bestMove = None

            for move in self.getLegalMoves():
                
        return 
    
    def getLegalMoves(self, myPos, advPos, maxStep, chessBoard):
        legalMoves = []
        moveQ = [(myPos, [], maxStep)]

        while moveQ:
            currentPos, legalMovesSoFar, stepsLeft = moveQ.pop(0)

            if stepsLeft == 0:
                legalMoves.append(legalMovesSoFar)
                continue

            # self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
            x, y = currentPos
            for direction in range(4):
                deltaX, deltaY = self.moves(direction)
                deltaPosition = (x + deltaX, y + deltaY)

                if not chessBoard[x, y, direction] and deltaPosition != advPos:
                    nextLegalMoves = legalMovesSoFar + [(deltaPosition, direction)]
                    moveQ.append((deltaPosition, nextLegalMoves, stepsLeft - 1))
        
        return legalMoves







             



#     def eval():
#         return 
    
#     def isGG():
#         return False
    
#     # Build a list of the moves we can make

# allowed_dirs = [ d                                
#     for d in range(0,4)                                      # 4 moves possible
#     if not self.chess_board[r,c,d] and                       # chess_board True means wall
#     not adv_pos == (r+self.moves[d][0],c+self.moves[d][1])]  # cannot move through Adversary

# # Moves (Up, Right, Down, Left)
#         self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

# def check_valid_step(self, start_pos, end_pos, barrier_dir):
#         """
#         Check if the step the agent takes is valid (reachable and within max steps).

#         Parameters
#         ----------
#         start_pos : tuple
#             The start position of the agent.
#         end_pos : np.ndarray
#             The end position of the agent.
#         barrier_dir : int
#             The direction of the barrier.
#         """
#         # Endpoint already has barrier or is border
#         r, c = end_pos
#         if self.chess_board[r, c, barrier_dir]:
#             return False
#         if np.array_equal(start_pos, end_pos):
#             return True

#         # Get position of the adversary
#         adv_pos = self.p0_pos if self.turn else self.p1_pos

#         # BFS
#         state_queue = [(start_pos, 0)]
#         visited = {tuple(start_pos)}
#         is_reached = False
#         while state_queue and not is_reached:
#             cur_pos, cur_step = state_queue.pop(0)
#             r, c = cur_pos
#             if cur_step == self.max_step:
#                 break
#             for dir, move in enumerate(self.moves):
#                 if self.chess_board[r, c, dir]:
#                     continue

#                 next_pos = cur_pos + move
#                 if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
#                     continue
#                 if np.array_equal(next_pos, end_pos):
#                     is_reached = True
#                     break

#                 visited.add(tuple(next_pos))
#                 state_queue.append((next_pos, cur_step + 1))

#         return is_reached

#     def check_endgame(self):
#         """
#         Check if the game ends and compute the current score of the agents.

#         Returns
#         -------
#         is_endgame : bool
#             Whether the game ends.
#         player_1_score : int
#             The score of player 1.
#         player_2_score : int
#             The score of player 2.
#         """
#         # Union-Find
#         father = dict()
#         for r in range(self.board_size):
#             for c in range(self.board_size):
#                 father[(r, c)] = (r, c)

#         def find(pos):
#             if father[pos] != pos:
#                 father[pos] = find(father[pos])
#             return father[pos]

#         def union(pos1, pos2):
#             father[pos1] = pos2

#         for r in range(self.board_size):
#             for c in range(self.board_size):
#                 for dir, move in enumerate(
#                     self.moves[1:3]
#                 ):  # Only check down and right
#                     if self.chess_board[r, c, dir + 1]:
#                         continue
#                     pos_a = find((r, c))
#                     pos_b = find((r + move[0], c + move[1]))
#                     if pos_a != pos_b:
#                         union(pos_a, pos_b)

#         for r in range(self.board_size):
#             for c in range(self.board_size):
#                 find((r, c))
#         p0_r = find(tuple(self.p0_pos))
#         p1_r = find(tuple(self.p1_pos))
#         p0_score = list(father.values()).count(p0_r)
#         p1_score = list(father.values()).count(p1_r)
#         if p0_r == p1_r:
#             return False, p0_score, p1_score
#         player_win = None
#         win_blocks = -1
#         if p0_score > p1_score:
#             player_win = 0
#             win_blocks = p0_score
#         elif p0_score < p1_score:
#             player_win = 1
#             win_blocks = p1_score
#         else:
#             player_win = -1  # Tie
#         if player_win >= 0:
#             logging.info(
#                 f"Game ends! Player {self.player_names[player_win]} wins having control over {win_blocks} blocks!"
#             )
#         else:
#             logging.info("Game ends! It is a Tie!")
#         return True, p0_score, p1_score

#     def check_boundary(self, pos):
#         r, c = pos
#         return 0 <= r < self.board_size and 0 <= c < self.board_size

#     def set_barrier(self, r, c, dir):
#         # Set the barrier to True
#         self.chess_board[r, c, dir] = True
#         # Set the opposite barrier to True
#         move = self.moves[dir]
#         self.chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

#     def random_walk(self, my_pos, adv_pos):
#         """
#         Randomly walk to the next position in the board.

#         Parameters
#         ----------
#         my_pos : tuple
#             The position of the agent.
#         adv_pos : tuple
#             The position of the adversary.
#         """
#         steps = np.random.randint(0, self.max_step + 1)

#         # Pick steps random but allowable moves
#         for _ in range(steps):
#             r, c = my_pos

#             # Build a list of the moves we can make
#             allowed_dirs = [ d                                
#                 for d in range(0,4)                                      # 4 moves possible
#                 if not self.chess_board[r,c,d] and                       # chess_board True means wall
#                 not adv_pos == (r+self.moves[d][0],c+self.moves[d][1])]  # cannot move through Adversary

#             if len(allowed_dirs)==0:
#                 # If no possible move, we must be enclosed by our Adversary
#                 break

#             random_dir = allowed_dirs[np.random.randint(0, len(allowed_dirs))]

#             # This is how to update a row,col by the entries in moves 
#             # to be consistent with game logic
#             m_r, m_c = self.moves[random_dir]
#             my_pos = (r + m_r, c + m_c)

#         # Final portion, pick where to put our new barrier, at random
#         r, c = my_pos
#         # Possibilities, any direction such that chess_board is False
#         allowed_barriers=[i for i in range(0,4) if not self.chess_board[r,c,i]]
#         # Sanity check, no way to be fully enclosed in a square, else game already ended
#         assert len(allowed_barriers)>=1 
#         dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]

#         return my_pos, dir

# def random_walk(self, my_pos, adv_pos):
#         """
#         Randomly walk to the next position in the board.

#         Parameters
#         ----------
#         my_pos : tuple
#             The position of the agent.
#         adv_pos : tuple
#             The position of the adversary.
#         """
#         steps = np.random.randint(0, self.max_step + 1)

#         # Pick steps random but allowable moves
#         for _ in range(steps):
#             r, c = my_pos

#             # Build a list of the moves we can make
#             allowed_dirs = [ d                                
#                 for d in range(0,4)                                      # 4 moves possible
#                 if not self.chess_board[r,c,d] and                       # chess_board True means wall
#                 not adv_pos == (r+self.moves[d][0],c+self.moves[d][1])]  # cannot move through Adversary

#             if len(allowed_dirs)==0:
#                 # If no possible move, we must be enclosed by our Adversary
#                 break

#             random_dir = allowed_dirs[np.random.randint(0, len(allowed_dirs))]

#             # This is how to update a row,col by the entries in moves 
#             # to be consistent with game logic
#             m_r, m_c = self.moves[random_dir]
#             my_pos = (r + m_r, c + m_c)

#         # Final portion, pick where to put our new barrier, at random
#         r, c = my_pos
#         # Possibilities, any direction such that chess_board is False
#         allowed_barriers=[i for i in range(0,4) if not self.chess_board[r,c,i]]
#         # Sanity check, no way to be fully enclosed in a square, else game already ended
#         assert len(allowed_barriers)>=1 
#         dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]

#         return my_pos, dir

# def check_boundary(self, pos):
#         r, c = pos
#         return 0 <= r < self.board_size and 0 <= c < self.board_size

#     def set_barrier(self, r, c, dir):
#         # Set the barrier to True
#         self.chess_board[r, c, dir] = True
#         # Set the opposite barrier to True
#         move = self.moves[dir]
#         self.chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

#     def random_walk(self, my_pos, adv_pos):
#         """
#         Randomly walk to the next position in the board.

#         Parameters
#         ----------
#         my_pos : tuple
#             The position of the agent.
#         adv_pos : tuple
#             The position of the adversary.
#         """
#         steps = np.random.randint(0, self.max_step + 1)

#         # Pick steps random but allowable moves
#         for _ in range(steps):
#             r, c = my_pos

#             # Build a list of the moves we can make
#             allowed_dirs = [ d                                
#                 for d in range(0,4)                                      # 4 moves possible
#                 if not self.chess_board[r,c,d] and                       # chess_board True means wall
#                 not adv_pos == (r+self.moves[d][0],c+self.moves[d][1])]  # cannot move through Adversary

#             if len(allowed_dirs)==0:
#                 # If no possible move, we must be enclosed by our Adversary
#                 break

#             random_dir = allowed_dirs[np.random.randint(0, len(allowed_dirs))]

#             # This is how to update a row,col by the entries in moves 
#             # to be consistent with game logic
#             m_r, m_c = self.moves[random_dir]
#             my_pos = (r + m_r, c + m_c)

#         # Final portion, pick where to put our new barrier, at random
#         r, c = my_pos
#         # Possibilities, any direction such that chess_board is False
#         allowed_barriers=[i for i in range(0,4) if not self.chess_board[r,c,i]]
#         # Sanity check, no way to be fully enclosed in a square, else game already ended
#         assert len(allowed_barriers)>=1 
#         dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]

#         return my_pos, dir