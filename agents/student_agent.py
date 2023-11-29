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
        self.transpositionTable = {}
        self.gameOverCache = {}

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
        # so far when it nears 2 seconds
        # 
        # Add iterative deepening
        # Move ordering
        # Transposition

        start_time = time.time()
        depth = 1
        bestMove = None

        while True:
            move, _ = self.alphaBeta(my_pos, adv_pos, depth, max_step, chess_board, True, float("-inf"), float("inf"), start_time)
            currentTime = time.time()

            if currentTime - start_time > 1.98:
                break  # Stop if we're close to the time limit

            bestMove = move  # Update best move at this depth
            depth += 1  # Increase depth for next iteration

        return bestMove
    
    def alphaBeta(self, myPos, advPos, depth, maxStep, chessBoard, isMaximizing, alpha, beta, startTime):
        # Using this as reference: http://people.csail.mit.edu/plaat/mtdf.html#abmem

        transpositionKey = self.getTranspositionKey(myPos, advPos, chessBoard, isMaximizing)
        
        # Check transposition table and update alpha and beta if possible
        if transpositionKey in self.transpositionTable:
            entry = self.transpositionTable[transpositionKey]
            if entry['depth'] >= depth:
                if entry['flag'] == 'EXACT':
                    return entry['bestMove'], entry['score']
                elif entry['flag'] == 'LOWERBOUND':
                    alpha = max(alpha, entry['score'])
                elif entry['flag'] == 'UPPERBOUND':
                    beta = min(beta, entry['score'])
                if alpha >= beta:
                    return entry['bestMove'], entry['score']
                    
        current_time = time.time()
        if current_time - startTime > 1.98 or depth == 0:
            score = self.eval(myPos, advPos, chessBoard, maxStep)
            return None, score

        if isMaximizing:
            maxScore = float("-inf")
            bestMove = None
            for move in self.getLegalMoves(myPos, advPos, maxStep, chessBoard):
                score = self.alphaBeta(move[0], advPos, depth - 1, maxStep, chessBoard, False, alpha, beta, startTime)[1]
                if score > maxScore:
                    maxScore = score
                    bestMove = move
                alpha = max(alpha, score)
                if beta <= alpha:
                    break

            # Save to transposition table
            flag = 'EXACT' if maxScore <= alpha else 'LOWERBOUND'
            self.transpositionTable[transpositionKey] = {'score': maxScore, 'depth': depth, 'bestMove': bestMove, 'flag': flag}
            return bestMove, maxScore
        else:
            minScore = float("inf")
            bestMove = None
            for move in self.getLegalMoves(advPos, myPos, maxStep, chessBoard):
                score = self.alphaBeta(myPos, move[0], depth - 1, maxStep, chessBoard, True, alpha, beta, startTime)[1]
                if score < minScore:
                    minScore = score
                    bestMove = move
                beta = min(beta, score)
                if beta <= alpha:
                    break

            # Save to transposition table
            flag = 'EXACT' if minScore >= beta else 'UPPERBOUND'
            self.transpositionTable[transpositionKey] = {'score': minScore, 'depth': depth, 'bestMove': bestMove, 'flag': flag}
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
    
    def getTranspositionKey(self, myPos, advPos, chessBoard, isMaximizing):
        return (myPos, advPos, chessBoard.tostring(), isMaximizing)

    def eval(self, myPos, advPos, chessBoard, maxStep):
        score = 0
        myMoves = len(self.getLegalMoves(myPos, advPos, maxStep, chessBoard))
        advMoves = len(self.getLegalMoves(advPos, myPos, maxStep, chessBoard))
        score += (myMoves - advMoves)

        # Distance from myPos to advPos
        # Distance to walls
        # Distance to all barriers
        # Gaps between closest barriers 
        
        return score

    def isGameOver(self, myPos, advPos, chessBoard):

        boardKey = chessBoard.tostring()
        if boardKey in self.gameOverCache:
            return self.gameOverCache[boardKey]

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
        gameOverResult = p0_r != p1_r
        self.gameOverCache[boardKey] = gameOverResult
        return gameOverResult