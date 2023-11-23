import numpy as np

class MockAgent:
    def checkBoundary(self, pos, boardLength):
        r, c = pos
        return 0 <= r < boardLength and 0 <= c < boardLength

    def getLegalMoves(self, myPos, advPos, maxStep, chessBoard):
        boardLength, _, _ = chessBoard.shape
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        legalMoves = []
        moveQ = [(myPos, [], maxStep)]

        while moveQ:
            currentPos, legalMovesSoFar, stepsLeft = moveQ.pop(0)

            if stepsLeft == 0:
                legalMoves.extend(legalMovesSoFar)
                continue


            x = currentPos[0]
            y = currentPos[1]

            for direction in range(4):
                deltaX = moves[direction][0]
                deltaY = moves[direction][1]
                deltaPosition = (x + deltaX, y + deltaY)

                if self.checkBoundary(deltaPosition, boardLength) and not chessBoard[x, y, direction] and deltaPosition != advPos:
                    nextLegalMoves = legalMovesSoFar + [(deltaPosition, direction)]
                    moveQ.append((deltaPosition, nextLegalMoves, stepsLeft - 1))
        
        return legalMoves

# Create a mock chess board with barriers and test the function
boardSize = 5
chessBoard = np.zeros((boardSize, boardSize, 4))
chessBoard[2, 2, :] = 1  # Place barriers around the cell (2, 2)

agent = MockAgent()
myPos = (1, 1)
advPos = (3, 3)
maxStep = 3

# Run the test
legalMoves = agent.getLegalMoves(myPos, advPos, maxStep, chessBoard)
print(legalMoves)
