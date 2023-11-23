import numpy as np

def print_chessboard(chessboard, player_pos, opponent_pos):
    """
    Converts the chessboard array into a human-readable string format.
    :param chessboard: A 3D numpy array representing the chessboard.
    :param player_pos: A tuple representing the player's position.
    :param opponent_pos: A tuple representing the opponent's position.
    :return: A string representation of the chessboard.
    """

    # Define symbols for barriers and corners
    horizontal_barrier = "---"
    vertical_barrier = "|"
    corner = "+"
    no_barrier = "   "
    player = " P "
    opponent = " O "
    
    rows, cols, _ = chessboard.shape
    board_str = ""

    for x in range(rows):
        # Add top horizontal barriers for the row
        board_str += corner
        for y in range(cols):
            board_str += horizontal_barrier if chessboard[x][y][0] else no_barrier
            board_str += corner
        board_str += "\n"

        # Add vertical barriers and player/opponent symbols
        for y in range(cols):
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
    board_str += corner
    for y in range(cols):
        board_str += horizontal_barrier if chessboard[rows - 1][y][2] else no_barrier
        board_str += corner

    return board_str

# Example usage
chessboard_example = np.array(
    [[[True, False, False, True], [True, False, True, False], [True, False, False, False], [True, False, False, False], [True, False, False, False], [True, True, False, False]],
     [[False, False, False, True], [True, False, True, False], [False, True, False, False], [False, True, False, True], [False, False, True, True], [False, True, False, False]],
     [[False, True, False, True], [True, False, False, True], [False, True, False, False], [False, False, False, True], [True, False, True, False], [False, True, False, False]],
     [[False, True, False, True], [False, False, False, True], [False, True, False, False], [False, False, False, True], [True, True, True, False], [False, True, False, True]],
     [[False, False, False, True], [False, True, False, False], [False, False, False, True], [False, False, True, False], [True, False, True, False], [False, True, False, False]],
     [[False, False, True, True], [False, False, True, False], [False, False, True, False], [True, False, True, False], [True, False, True, False], [False, True, True, False]]]
)

print(print_chessboard(chessboard_example, (3,3), (1,1)))
