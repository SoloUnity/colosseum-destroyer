# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from collections import deque
import math
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
        bestScore = 0
        bestMove = None

        while True:
            score, move = self.MaxValue(my_pos, adv_pos, depth, max_step, chess_board, float("-inf"), float("inf"), start_time)
            currentTime = time.time()
            
            if score > bestScore:
                bestMove = move  # Update best move at this depth
                bestScore = score

            if currentTime - start_time > 1.98:
                break  # Stop if we're close to the time limit

            depth += 1  # Increase depth for next iteration

        print("depth: " + str(depth))
        return bestMove
    
    # Using this as reference: http://people.csail.mit.edu/plaat/mtdf.html#abmem
    def MaxValue(self, myPos, advPos, depth, maxStep, chessBoard, alpha, beta, startTime):
        if self.cutoff(myPos, advPos, depth, chessBoard, startTime):
            return self.eval(myPos, advPos, chessBoard, maxStep), None

        #self.checkTable(myPos, advPos, chessBoard, depth)

        maxScore = float("-inf")
        bestMove = None
        for move in self.getLegalMoves(myPos, advPos, maxStep, chessBoard):
            score, _ = self.MinValue(move[0], advPos, depth - 1, maxStep, chessBoard, alpha, beta, startTime)
            if score > maxScore:
                maxScore = score
                bestMove = move
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return maxScore, bestMove

    def MinValue(self, myPos, advPos, depth, maxStep, chessBoard, alpha, beta, startTime):
        if self.cutoff(myPos, advPos, depth, chessBoard, startTime):
            return self.eval(myPos, advPos, chessBoard, maxStep), None

        #self.checkTable(myPos, advPos, chessBoard, depth)

        minScore = float("inf")
        bestMove = None
        for move in self.getLegalMoves(advPos, myPos, maxStep, chessBoard):
            score, _ = self.MaxValue(myPos, move[0], depth - 1, maxStep, chessBoard, alpha, beta, startTime)
            if score < minScore:
                minScore = score
                bestMove = move
            beta = min(beta, score)
            if alpha >= beta:
                return minScore, bestMove
        return minScore, bestMove

    def cutoff(self, myPos, advPos, depth, chessBoard, startTime):
        current_time = time.time()
        return current_time - startTime > 1.98 or depth == 0

        # return current_time - startTime > 1.98 or depth == 0 or self.isGameOver(myPos, advPos, chessBoard)
    
    def checkTable(self, myPos, advPos, chessBoard, isMaximizing, depth):
        transpositionKey = self.getTranspositionKey(myPos, advPos, chessBoard, True)
        if transpositionKey in self.transpositionTable:
            lowerbound, upperbound = self.transpositionTable[transpositionKey]
            if lowerbound >= beta:
                return lowerbound, None
            if upperbound <= alpha:
                return upperbound, None
            alpha = max(alpha, lowerbound)
            beta = min(beta, upperbound)
                    
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
        boardLength, _, _ = chessBoard.shape
        centerPos = boardLength / 2
        myDist = math.sqrt(((myPos[0] + 1 - centerPos) ** 2) + ((myPos[1] + 1 - centerPos) ** 2))
        advDist = math.sqrt(((advPos[0] + 1 - centerPos) ** 2) + ((advPos[1] + 1 - centerPos) ** 2))

        # myMoves = len(self.getLegalMoves(myPos, advPos, maxStep, chessBoard))
        # advMoves = len(self.getLegalMoves(advPos, myPos, maxStep, chessBoard))
        score += (advDist - myDist)

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