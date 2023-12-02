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
                break  # Stop if we"re close to the time limit

            depth += 1  # Increase depth for next iteration

        print("depth: " + str(depth))
        return bestMove
    
    # Using this as reference: http://people.csail.mit.edu/plaat/mtdf.html#abmem
    def MaxValue(self, myPos, advPos, depth, maxStep, chessBoard, alpha, beta, startTime):
        if self.cutoff(myPos, advPos, depth, chessBoard, startTime):
            return self.eval(myPos, advPos, chessBoard, maxStep), None

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
        for move in self.getLegalMoves(myPos, advPos, maxStep, chessBoard):
            score, _ = self.MinValue(move[0], advPos, depth - 1, maxStep, chessBoard, alpha, beta, startTime)
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

    def MinValue(self, myPos, advPos, depth, maxStep, chessBoard, alpha, beta, startTime):
        if self.cutoff(myPos, advPos, depth, chessBoard, startTime):
            return self.eval(myPos, advPos, chessBoard, maxStep), None

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
        for move in self.getLegalMoves(advPos, myPos, maxStep, chessBoard):
            score, _ = self.MaxValue(myPos, move[0], depth - 1, maxStep, chessBoard, alpha, beta, startTime)
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

    def cutoff(self, myPos, advPos, depth, chessBoard, startTime):
        current_time = time.time()
        return current_time - startTime > 1.98 or depth == 0

        # return current_time - startTime > 1.98 or depth == 0 or self.isGameOver(myPos, advPos, chessBoard)
    

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
        control = self.zoneArea(myPos, advPos, chessBoard)

        score = (self.potentialExpansion(myPos, advPos, maxStep, chessBoard) * 0.3
        + self.distanceFromCenter(myPos, advPos, chessBoard) * 0.1
        + control * 0.2
        + self.zoneBarriers(myPos, advPos, chessBoard, control) * 0.3)

        return score

    def potentialExpansion(self, myPos, advPos, maxStep, chessBoard):
        myMoves = len(self.getLegalMoves(myPos, advPos, maxStep, chessBoard))
        advMoves = len(self.getLegalMoves(advPos, myPos, maxStep, chessBoard))
        return (myMoves - advMoves)
    
    def distanceFromCenter(self, myPos, advPos, chessBoard):
        centerPos = (chessBoard.shape[0] - 1) / 2
        myDist = abs(myPos[0] - centerPos) + abs(myPos[1] - centerPos)
        advDist = abs(advPos[0] - centerPos) + abs(advPos[1] - centerPos)
        return (advDist - myDist)
    
    def zoneArea(self, myPos, advPos, chessBoard):
        if myPos[0] > advPos[0]:
            midPosX = (myPos[0] - advPos[0]) / 2 + advPos[0]
        else:
            midPosX = (advPos[0] - myPos[0]) / 2 + myPos[0]

        if myPos[1] > advPos[1]:
            midPosY = (myPos[1] - advPos[1]) / 2 + advPos[1]
        else:
            midPosY = (advPos[1] - myPos[1]) / 2 + myPos[1]

        myZone = None
        advZone = None

        # Determine the area of the players' zones
        if myPos[0] == advPos[0]:
            if myPos[1] > midPosY:
                myZone = 6    
                advZone = 5
            else:
                myZone = 5
                advZone = 6

        elif myPos[1] == advPos[1]:
            if myPos[0] > midPosX:
                myZone = 8
                advZone = 7
            else:
                myZone = 7
                advZone = 8
            
        elif myPos[0] > midPosX:
            if myPos[1] > midPosY:
                myZone = 4
                advZone = 2
            else:
                myZone = 3
                advZone = 1

        else:
            if myPos[1] > midPosY:
                myZone = 1
                advZone = 3

            else:
                myZone = 2
                advZone = 4
                
        myMap = self.determineZones(chessBoard, myZone, midPosX, midPosY)
        advMap = self.determineZones(chessBoard, advZone, midPosX, midPosY)
        myArea = (math.floor(myMap[1]) - math.ceil(myMap[0]) + 1) * (math.floor(myMap[3]) - math.ceil(myMap[2]) + 1)
        advArea = (math.floor(advMap[1]) - math.ceil(advMap[0]) + 1) * (math.floor(advMap[3]) - math.ceil(advMap[2]) + 1)

        return (myArea - advArea)


    def determineZones(self, chessBoard, zone, midPosX, midPosY):
        startX, endX, startY, endY = 0, 0, 0, 0
        boardLength, _, _ = chessBoard.shape
        boardLength -= 1

        if zone == 1:   # Top-left quadrant
            startX = 0
            endX = midPosX
            startY = midPosY
            endY = boardLength
        
        elif zone == 2: # Top-right quadrant
            startX = 0
            endX = midPosX
            startY = 0
            endY = midPosY
        
        elif zone == 3: # Bottom-right quadrant
            startX = midPosX
            endX = boardLength
            startY = 0
            endY = midPosY

        elif zone == 4: # Bottom-left quadrant
            startX = midPosX
            endX = boardLength
            startY = midPosY
            endY = boardLength

        elif zone == 5: # Left half
            startX = 0
            endX = boardLength
            startY = 0
            endY = midPosY
      
        elif zone == 6: # Right half
            startX = 0
            endX = boardLength
            startY = midPosY
            endY = boardLength

        elif zone == 7: # Top half
            startX = 0
            endX = midPosX
            startY = 0
            endY = boardLength

        elif zone == 8: # Bottom half
            startX = midPosX
            endX = boardLength
            startY = 0
            endY = boardLength
            
        return (startX, endX, startY, endY)
    
    def zoneBarriers(self, myPos, advPos, chessBoard, control):
        if control == 0:
            return 0
        
        boardLength, _, _ = chessBoard.shape
        count = 0

        if myPos[0] > advPos[0]:
            startX = advPos[0]
            endX = myPos[0]
        else:
            startX = myPos[0]
            endX = advPos[0]

        if myPos[1] > advPos[1]:
            startY = advPos[1]
            endY = myPos[1]
        else:
            startY = myPos[1]
            endY = advPos[1]

        for r in range(startX, endX):
            for c in range(startY, endY):
                if chessBoard[r, c, 1]:
                    count += 1
                if chessBoard[r, c, 2]:
                    count += 1
    
        if control > 0:
            return count

        else: 
            return -count
    
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