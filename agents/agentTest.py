import numpy as np

class MockAgent:
    def checkBoundary(self, pos, boardLength):
        r, c = pos
        return 0 <= r < boardLength and 0 <= c < boardLength

    def getLegalMoves1(self, myPos, advPos, maxStep, chessBoard):
        boardLength, _, _ = chessBoard.shape
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        legalMoves = []
        moveQ = [(myPos, maxStep)]

        while moveQ:
            currentPos, stepsLeft = moveQ.pop(0)

            x, y = currentPos

            # Add staying in place with barrier placement
            for direction in range(4):
                if not chessBoard[x, y, direction]:  # Check for no barrier
                    legalMoves.append((currentPos, direction))

            # Check for legal moves and barrier placement
            if stepsLeft > 0:
                for direction in range(4):
                    deltaX, deltaY = moves[direction]
                    deltaPosition = (x + deltaX, y + deltaY)

                    # Check for legal movement
                    if self.checkBoundary(deltaPosition, boardLength) and deltaPosition != advPos:
                        if not chessBoard[x, y, direction]:  # Ensure no barrier in the direction of movement
                            # Add barrier placement for the new position
                            for barrierDir in range(4):
                                if not chessBoard[deltaPosition[0], deltaPosition[1], barrierDir]:  # Check for no barrier
                                    legalMoves.append((deltaPosition, barrierDir))

        # Remove duplicates from legalMoves
        legalMoves = list(set(legalMoves))

        return legalMoves
    
    def getLegalMoves2(self, myPos, advPos, maxStep, chessBoard):
        boardLength, _, _ = chessBoard.shape
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        legalMoves = set()
        visited = set()
        queue = [(myPos, maxStep)]

        # Case for surrounding squares
        # BFS
        while queue:
            currentPos, stepsLeft = queue.pop(0)
            visited.add(currentPos)
            
            for directionIndex in range(4):

                boold = chessBoard[currentPos[0]][currentPos[1]][directionIndex]
                print("Boolean: " + str(boold) + " Position: " + str((currentPos[0],currentPos[1], directionIndex)))
                if not(chessBoard[currentPos[0]][currentPos[1]][directionIndex]):
                    #print("Adding: " + str((currentPos, directionIndex)))
                    legalMoves.add((currentPos, directionIndex))

            if stepsLeft > 0:
                for directionIndex in range(4):

                    deltaX = currentPos[0] + moves[directionIndex][0]
                    deltaY = currentPos[1] + moves[directionIndex][1]
                    nextPosition = (deltaX, deltaY)

                    if (nextPosition not in visited) and (self.checkBoundary(nextPosition, boardLength)) and (nextPosition != advPos[0]):
                        queue.append((nextPosition, stepsLeft - 1))

        return list(legalMoves)











def print_chessboard(chessboard, player_pos, opponent_pos):
    """
    Converts the chessboard array into a human-readable string format.
    :param chessboard: A 3D numpy array representing the chessboard.
    :param player_pos: A tuple representing the player's position.
    :param opponent_pos: A tuple representing the opponent's position.
    :return: A string representation of the chessboard.
    """

    horizontal_barrier = "---"
    vertical_barrier = "|"
    corner = "+"
    no_barrier = "   "
    player = " P "
    opponent = " O "

    rows, cols, _ = chessboard.shape
    board_str = "    "  # Initial spacing for alignment

    # Print top indexes
    for y in range(cols):
        board_str += f" {y}  "
    board_str += "\n"

    for x in range(rows):
        # Add top horizontal barriers for the row
        board_str += "  " + corner  # Align with the columns
        for y in range(cols):

            boold = chessboard[x][y][0]
            # print("Boolean: " + str(boold) + " Position: " + str(((x,y), 0)))

            board_str += horizontal_barrier if chessboard[x][y][0] else no_barrier
            board_str += corner
        board_str += "\n"

        # Add vertical barriers and player/opponent symbols
        board_str += f"{x} " if x < 10 else f"{x}"  # Align single and double-digit numbers
        for y in range(cols):

            boold = chessboard[x][y - 1][1]
            # print("Boolean: " + str(boold) + " Position: " + str(((x,y-1), 1)))

            if y == 0 or chessboard[x][y - 1][1]:
                board_str += vertical_barrier
            else:
                board_str += " "

            if (x, y) == player_pos:
                board_str += player
            elif (x, y) == opponent_pos:
                board_str += opponent
            else:
                board_str += "   "

        board_str += vertical_barrier  # Rightmost vertical barrier
        board_str += "\n"

    # Add bottom horizontal barriers for the last row
    board_str += "  " + corner
    for y in range(cols):

        boold = chessboard[rows - 1][y][2]
        # print("Boolean: " + str(boold) + " Position: " + str(((rows - 1,y-1), 2)))
        
        board_str += horizontal_barrier if chessboard[rows - 1][y][2] else no_barrier
        board_str += corner

    return board_str

# Create a mock chess board with barriers and test the function
boardSize = 6
data = [
  [
    [True, False, False, True],
    [True, True, False, False],
    [True, False, False, True],
    [True, False, False, False],
    [True, False, False, False],
    [True, True, False, False]
  ],
  [
    [False, False, False, True],
    [False, False, False, False],
    [False, False, False, False],
    [False, False, True, False],
    [False, False, False, False],
    [False, True, False, False]
  ],
  [
    [False, False, False, True],
    [False, False, False, False],
    [False, False, False, False],
    [True, False, False, False],
    [False, False, False, False],
    [False, True, False, False]
  ],
  [
    [False, False, False, True],
    [False, False, False, False],
    [False, False, True, False],
    [False, False, False, False],
    [False, False, False, False],
    [False, True, False, False]
  ],
  [
    [False, False, False, True],
    [False, False, False, False],
    [True, False, False, False],
    [False, False, False, False],
    [False, False, False, False],
    [False, True, False, False]
  ],
  [
    [False, False, True, True],
    [False, False, True, False],
    [False, False, True, False],
    [False, True, True, False],
    [False, False, True, True],
    [False, True, True, False]
  ]
]

# Initialize the chessboard with False (equivalent to 0) values
chessBoard = np.array(data, dtype=bool)


# # Place barriers around the cell (2, 2) by setting them to True
# barrier_positions = [(2, 1), (2, 2), (2, 3), (1, 2), (3, 2)]
# for pos in barrier_positions:
#     chessBoard[pos] = True

agent = MockAgent()
myPos = (1, 1)
advPos = (3, 3)
maxStep = 1

print(print_chessboard(chessBoard, myPos, advPos))
# Run the test
legalMoves = agent.getLegalMoves2(myPos, advPos, maxStep, chessBoard)
for i in sorted(legalMoves):
    print (i)

# Reference answer is: 
# [ 
#   ((0, 1), 2), 
#   ((0, 1), 3),
#   ((1, 0), 2), 
#   ((1, 0), 3), 
#   ((1, 1), 0),
#   ((1, 1), 1), 
#   ((1, 1), 2), 
#   ((1, 1), 3) 
#   ((1, 2), 0),    
#   ((1, 2), 1), 
#   ((1, 2), 2), 
#   ((1, 2), 3),   
#   ((2, 1), 0), 
#   ((2, 1), 1), 
#   ((2, 1), 2), 
#   ((2, 1), 3), 
#   
# ]

# self.dir_map = {
#             "u": 0,
#             "r": 1,
#             "d": 2,
#             "l": 3,
#         }
