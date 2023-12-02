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
        score = (self.potentialExpansion(myPos, advPos, maxStep, chessBoard) * 0
        + self.distanceFromCenter(myPos, advPos, chessBoard) * 1
        + self.longestLineZone(myPos, advPos, chessBoard) * 0)

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
    
    def longestLineZone(self, myPos, advPos, chessBoard):
        newBoard = deepcopy(chessBoard)
        boardLength, _, _ = newBoard.shape
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        longestLine = self.findLongestLine(myPos, advPos, newBoard, boardLength, moves)

        # No barriers on the board
        if any(line is None for line in longestLine):
            return 0
        
        self.extendLine(longestLine[0], newBoard, boardLength, moves)
        self.extendLine(longestLine[1], newBoard, boardLength, moves)

        myScore, advScore = self.computeScore(myPos, advPos, newBoard, boardLength, moves)

        return (myScore - advScore)
    
    # def findLongestLine(self, myPos, advPos, chessBoard, boardLength, moves):
    #     # Find longest line
    #     # for every barrier that is not a wall (only check down and right)
    #         # dfs to find longest line: 3 to check at every step
    #         # keep track of visited barriers
    #     visited = set()
    #     longest_line = []

    #     for r in range(boardLength):
    #         for c in range(boardLength):
    #             if (r,c,1) not in visited:
    #                 if chessBoard[r, c, 1]:
    #                     local_line = [(r, c, 1)]
    #                 elif chessBoard[r, c, 2]:
    #                     local_line = [(r, c, 2)]
    #                 else:
    #                     continue
                    
    #                 # check right cell down
    #                 # check down cell right

    #                 self.barrierDFS(r, c, visited, local_line)

                        
        
        

    #     return longest
    
    def barrierDFS(r, c, visited, barrier):
        pass
    
    def extendLine(self, line, chessBoard, boardLength, moves):
            # for first and last barrier of line
            # use min from wall of x,y coord and extend in that direction
        return chessBoard
    
    def computeScore(self, myPos, advPos, chessBoard, boardLength, moves):
        # Count number of squares in each zone

        # # Union-Find
        # father = dict()
        # for r in range(self.board_size):
        #     for c in range(self.board_size):
        #         father[(r, c)] = (r, c)

        # def find(pos):
        #     if father[pos] != pos:
        #         father[pos] = find(father[pos])
        #     return father[pos]

        # def union(pos1, pos2):
        #     father[pos1] = pos2

        # for r in range(self.board_size):
        #     for c in range(self.board_size):
        #         for dir, move in enumerate(
        #             self.moves[1:3]
        #         ):  # Only check down and right
        #             if self.chess_board[r, c, dir + 1]:
        #                 continue
        #             pos_a = find((r, c))
        #             pos_b = find((r + move[0], c + move[1]))
        #             if pos_a != pos_b:
        #                 union(pos_a, pos_b)

        # for r in range(self.board_size):
        #     for c in range(self.board_size):
        #         find((r, c))
        # p0_r = find(tuple(self.p0_pos))
        # p1_r = find(tuple(self.p1_pos))
        # p0_score = list(father.values()).count(p0_r)
        # p1_score = list(father.values()).count(p1_r)
        # if p0_r == p1_r:
        #     return False, p0_score, p1_score
        # player_win = None
        # win_blocks = -1
        # if p0_score > p1_score:
        #     player_win = 0
        #     win_blocks = p0_score
        # elif p0_score < p1_score:
        #     player_win = 1
        #     win_blocks = p1_score
        # else:
        #     player_win = -1  # Tie
        # if player_win >= 0:
        #     logging.info(
        #         f"Game ends! Player {self.player_names[player_win]} wins having control over {win_blocks} blocks!"
        #     )
        # else:
        #     logging.info("Game ends! It is a Tie!")
        # return True, p0_score, p1_score

        
        # return (myScore, advScore) 
        return None
    
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