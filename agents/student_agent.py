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

    def __init__(self, expansionWeight=1, agressiveWeight=1, centerDistanceWeight=1, aggressionWeight=1, midGameWeight=1, survivalWeight=1, openSpaceWeight=1, extendBarrierWeight=1):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

        self.maxStep = 0
        self.cutoffTime = 1.98
        self.gameOverThreshold = 20

        self.expansionWeight = expansionWeight
        self.agressiveWeight = agressiveWeight
        self.centerDistanceWeight = centerDistanceWeight
        self.aggressionWeight = aggressionWeight
        self.midGameWeight = midGameWeight
        self.survivalWeight = survivalWeight
        self.openSpaceWeight = openSpaceWeight
        self.extendBarrierWeight = extendBarrierWeight

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
        where (x, y) is the next pos of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds
        # 
        # Add iterative deepening
        # Move ordering
        # Transpos
        self.maxStep = max_step
        start_time = time.time()
        depth = 1
        bestScore = float("-inf")
        bestMove = None

        while True:
            score, move = self.alphaBeta(my_pos, adv_pos, depth, max_step, chess_board, start_time)
            currentTime = time.time()
            
            if score > bestScore:
                bestMove = move  # Update best move at this depth
                bestScore = score

            if currentTime - start_time > self.cutoffTime:
                break  # Stop if we"re close to the time limit

            depth += 1  # Increase depth for next iteration

        print("Depth: " + str(depth))

        return bestMove
    
    def alphaBeta(self, myPos, advPos, depth, maxStep, chessBoard, startTime):
        alpha = float("-inf")
        beta = float("inf")
        return self.MaxValue(myPos, advPos, depth, maxStep, chessBoard, alpha, beta, startTime)

    # Using this as reference: http://people.csail.mit.edu/plaat/mtdf.html#abmem
    def MaxValue(self, myPos, advPos, depth, maxStep, chessBoard, alpha, beta, startTime, evalMove = None):
        if self.cutoff(myPos, advPos, depth, chessBoard, startTime, evalMove):
            return self.eval(myPos, advPos, chessBoard, maxStep, evalMove), None

        transpositionKey = self.getTranspositionKey(myPos, advPos, chessBoard, True)
        if transpositionKey in self.transpositionTable:
            entry = self.transpositionTable[transpositionKey]
            if entry["depth"] >= depth:
                if entry["flag"] == "EXACT":
                    return entry["score"], entry["bestMove"]
                elif entry["flag"] == "LOWERBOUND":
                    alpha = max(alpha, entry["score"])
                elif entry["flag"] == "UPPERBOUND":
                    beta = min(beta, entry["score"])
                if alpha >= beta:
                    return entry["score"], entry["bestMove"]

        maxScore = float("-inf")
        bestMove = None
        legalMoves = self.getLegalMoves(myPos, advPos, maxStep, chessBoard)
        # sortedMoves = sorted(legalMoves, key=lambda move: self.eval(myPos, advPos, chessBoard, maxStep, move), reverse=True)

        for move in legalMoves:
            score, _ = self.MinValue(move[0], advPos, depth - 1, maxStep, chessBoard, alpha, beta, startTime, move)
            if score > maxScore:
                maxScore = score
                bestMove = move
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        # Save to transposition table
        flag = "EXACT" if maxScore <= alpha else "LOWERBOUND"
        self.transpositionTable[transpositionKey] = {"score": maxScore, "depth": depth, "bestMove": bestMove, "flag": flag}
        
        return maxScore, bestMove

    def MinValue(self, myPos, advPos, depth, maxStep, chessBoard, alpha, beta, startTime, evalMove=None):
        if self.cutoff(myPos, advPos, depth, chessBoard, startTime, evalMove):
            return self.eval(myPos, advPos, chessBoard, maxStep, evalMove), None

        transpositionKey = self.getTranspositionKey(myPos, advPos, chessBoard, False)
        if transpositionKey in self.transpositionTable:
            entry = self.transpositionTable[transpositionKey]
            if entry["depth"] >= depth:
                if entry["flag"] == "EXACT":
                    return entry["score"], entry["bestMove"]
                elif entry["flag"] == "LOWERBOUND":
                    alpha = max(alpha, entry["score"])
                elif entry["flag"] == "UPPERBOUND":
                    beta = min(beta, entry["score"])
                if alpha >= beta:
                    return entry["score"], entry["bestMove"]

        minScore = float("inf")
        bestMove = None

        legalMoves = self.getLegalMoves(advPos, myPos, maxStep, chessBoard)
        # sortedMoves = sorted(legalMoves, key=lambda move: self.eval(myPos, advPos, chessBoard, maxStep, move), reverse=False)

        for move in legalMoves:
            score, _ = self.MaxValue(myPos, move[0], depth - 1, maxStep, chessBoard, alpha, beta, startTime, move)
            if score < minScore:
                minScore = score
                bestMove = move
            beta = min(beta, score)
            if alpha >= beta:
                return minScore, bestMove
            
        # Save to transposition table
        flag = "EXACT" if minScore >= beta else "UPPERBOUND"
        self.transpositionTable[transpositionKey] = {"score": minScore, "depth": depth, "bestMove": bestMove, "flag": flag}

        return minScore, bestMove

    def cutoff(self, myPos, advPos, depth, chessBoard, startTime, evalMove):
        current_time = time.time()

        return current_time - startTime > self.cutoffTime or depth == 0
        # gameOver = False

        # if depth >= self.gameOverThreshold and self.isGameOver(myPos, advPos, chessBoard):
        #     gameOver = True

        # if evalMove == None:
        #     return False
        # else:
        #     return current_time - startTime > self.cutoffTime or depth == 0 or gameOver

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
    
    def eval(self,myPos, advPos, chessBoard, maxStep, move=None):
        score = 0
        boardSize = chessBoard.shape[0]
        totalWalls = np.sum(chessBoard)
        
        score += self.expansionHeuristic(myPos, advPos, maxStep, chessBoard) * self.expansionWeight
        score += self.aggressiveHeuristic(myPos, advPos, boardSize, totalWalls) * self.agressiveWeight
        score += self.centerDistanceHeuristic(chessBoard, myPos) * self.centerDistanceWeight
        score += self.openSpaceHeuristic(chessBoard, myPos) * self.openSpaceWeight
        if move != None:
            score += self.boxedHeuristic(myPos, advPos, chessBoard, move)
            score += self.extendBarrierHeuristic (move, chessBoard) * self.extendBarrierWeight
        return score

    def boxedHeuristic(self, myPos, advPos, chessBoard, move):
        x = myPos[0]
        y = myPos[1]
        barrierCount = 0

        if move == ((x, y), 0) or chessBoard[x, y, 0]:  # up
            barrierCount += 1
        if move == ((x, y), 1) or chessBoard[x, y, 1]:  # right
            barrierCount += 1
        if move == ((x, y), 2) or (y < chessBoard.shape[0] - 1 and chessBoard[x, y + 1, 2]):  # down
            barrierCount += 1
        if move == ((x, y), 3) or (x < chessBoard.shape[1] - 1 and chessBoard[x + 1, y, 3]):  # left
            barrierCount += 1

        manhattan_distance = abs(myPos[0] - advPos[0]) + abs(myPos[1] - advPos[1])

        if barrierCount == 3 and manhattan_distance <= self.maxStep - 1:
            return -9999
        elif barrierCount == 4:
            return -9999  

        return 0

        
    def expansionHeuristic(self, myPos, advPos, maxStep, chessBoard):
        myMoves = len(self.getLegalMoves(myPos, advPos, maxStep, chessBoard))
        advMoves = len(self.getLegalMoves(advPos, myPos, maxStep, chessBoard))
        return (myMoves - advMoves)
    
    def aggressiveHeuristic(self, myPos, advPos, boardSize, totalWalls):
        distanceToAdv = abs(myPos[0] - advPos[0]) + abs(myPos[1] - advPos[1])
        score = 0
        maxWalls = boardSize * (boardSize - 1) * 2  # Maximum possible walls

        # Calculate the percentage of walls placed
        wallPercentage = (totalWalls / maxWalls) * 100

        # Early game: Less than 33% of walls are placed
        if wallPercentage < 33:
            score -= distanceToAdv * self.aggressionWeight  # Be aggressive

        # Mid game: Between 33% and 66% of walls are placed
        elif wallPercentage < 66:
            score = -distanceToAdv * self.midGameWeight  # Adjust mid-game strategy

        # Late game: More than 66% of walls are placed
        else:
            score += distanceToAdv * self.survivalWeight  # Focus on survival
        return score
    
    def centerDistanceHeuristic(self,chessBoard, myPos):
        x, y = myPos
        centerPos = len(chessBoard) / 2
        return -(abs(x - centerPos) + abs(y - centerPos))

    def openSpaceHeuristic(self,chessBoard, myPos):
        x, y = myPos
        totalSquares = 0
        totalWalls = 0
        for x in range(max(0, x - 1), min(len(chessBoard), x + 1)):
            for y in range(max(0, y - 1), min(len(chessBoard), y + 1)):
                totalSquares += 1
                for direction in range(0, 4):
                    if chessBoard[x, y, direction]:
                        totalWalls += 1
        return (totalWalls - totalWalls) / totalSquares
    
    def extendBarrierHeuristic(self, move, chessBoard):
        score = 0
        x, y, direction = move[0][0], move[0][1], move[1]

        if direction == 0 or direction == 2:  # vertical move 
            if (y > 0 and chessBoard[x, y - 1, 1]) or (y < chessBoard.shape[1] - 1 and chessBoard[x, y + 1, 3]):
                score += 1
        elif direction == 1 or direction == 3:  # horizontal move 
            if (x > 0 and chessBoard[x - 1, y, 2]) or (x < chessBoard.shape[0] - 1 and chessBoard[x + 1, y, 0]):
                score += 1

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


    
   