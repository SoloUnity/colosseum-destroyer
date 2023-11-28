# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from collections import deque

import logging
logger = logging.getLogger(__name__)

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
        self.cachedLegalMoves = {}


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
        depth = 4 # Variable depth needed
        move, _ = self.alphaBeta(my_pos, adv_pos, depth, max_step, chess_board, True, float("-inf"), float("inf"), start_time)
        
        return move
    
    def alphaBeta(self, myPos, advPos, depth, maxStep, chessBoard, isMaximizing, alpha, beta, startTime):
        current_time = time.time()
        if current_time - startTime > 1.95 or depth == 0:
            return None, self.eval(myPos, advPos, chessBoard, maxStep)
        
        if isMaximizing:
            maxScore = float("-inf")
            bestMove = None
            for move in self.getLegalMoves(myPos, advPos, maxStep, chessBoard):
                score = (self.alphaBeta(move[0], advPos, depth - 1, maxStep, chessBoard, False, alpha, beta, startTime))[1]
                if score > maxScore:
                    maxScore = score
                    bestMove = move
                alpha = max(alpha, score)
                if beta <= alpha:
                    break

            return bestMove, maxScore
        else:

            minScore = float("inf")
            bestMove = None
            for move in self.getLegalMoves(advPos, myPos, maxStep, chessBoard):
                score = (self.alphaBeta(myPos, move[0], depth - 1, maxStep, chessBoard, True, alpha, beta, startTime))[1]
                if score < minScore:
                    minScore = score
                    bestMove = move
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return bestMove, minScore
                    
    # Needs optimization
    def getLegalMoves(self, myPos, advPos, maxStep, chessBoard):

        key = (myPos, advPos, maxStep, chessBoard.tostring())

        if key in self.cachedLegalMoves:
            return self.cachedLegalMoves[key]
        
        boardLength, _, _ = chessBoard.shape
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        legalMoves = set()
        visited = set()
        queue = deque([(myPos, maxStep)])

        while queue:
            currentPos, stepsLeft = queue.popleft()

            if currentPos in visited or not self.checkBoundary(currentPos, boardLength) or currentPos == advPos:
                continue
            visited.add(currentPos)

            for directionIndex, (deltaX, deltaY) in enumerate(moves):
                if not chessBoard[currentPos[0]][currentPos[1]][directionIndex]:
                    nextPosition = (currentPos[0] + deltaX, currentPos[1] + deltaY)
                    if nextPosition not in visited:
                        if stepsLeft > 0:
                            queue.append((nextPosition, stepsLeft - 1))
                    legalMoves.add((currentPos, directionIndex))

        self.cachedLegalMoves[key] = list(legalMoves)
        return list(legalMoves)

    def checkBoundary(self, pos, boardSize):
        x, y = pos
        return 0 <= x < boardSize and 0 <= y < boardSize
    
    def eval(self, myPos, advPos, chessBoard, maxStep):
        score = 0
        myMoves = len(self.getLegalMoves(myPos, advPos, maxStep, chessBoard))
        advMoves = len(self.getLegalMoves(advPos, myPos, maxStep, chessBoard))
        score += (myMoves - advMoves)
        return score

    def isGameOver(self, myPos, advPos, chessBoard):
        boardLength, _, _ = chessBoard.shape

        # Union-Find
        father = dict()
        for r in range(boardLength):
            for c in range(boardLength):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[find(pos1)] = find(pos2)

        # Only check down and right
        directions = [(0, 1), (1, 0)]  # Right, Down
        for r in range(boardLength):
            for c in range(boardLength):
                for move in directions:
                    if chessBoard[r, c, 1 if move == (0, 1) else 2]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        # Find roots for each player
        p0_r = find(tuple(myPos))
        p1_r = find(tuple(advPos))

        # Check if players belong to the same set
        return p0_r != p1_r





    def check_endgame(self):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        # Union-Find
        father = dict()
        for r in range(self.board_size):
            for c in range(self.board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(self.board_size):
            for c in range(self.board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if self.chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(self.board_size):
            for c in range(self.board_size):
                find((r, c))
        p0_r = find(tuple(self.p0_pos))
        p1_r = find(tuple(self.p1_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
        if player_win >= 0:
            logging.info(
                f"Game ends! Player {self.player_names[player_win]} wins having control over {win_blocks} blocks!"
            )
        else:
            logging.info("Game ends! It is a Tie!")
        return True, p0_score, p1_score

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
#         # Sanity check, no way to be fully enclosed in a square, else game alreadeltaY ended
#         assert len(allowed_barriers)>=1 
#         dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]

#         return my_pos, dir
