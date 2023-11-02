"""
Tic Tac Toe Player
"""
import math
from copy import deepcopy
import random
X = "X"
O = "O"
EMPTY = None

def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]

def player(board):
    """
    Returns player who has the next turn on a board.
    """
    Number_of_X = sum(row.count(X) for row in board)
    Number_of_O = sum(row.count(O) for row in board)

    if Number_of_X > Number_of_O:
        return O
    else:
        return X

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    available_moves = {(i, j) for i in range(3) for j in range(3) if board[i][j] == EMPTY}
    return available_moves
def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action not in actions(board):
        raise Exception("This action is not valid")

    i, j = action

    board_copy = deepcopy(board)
    board_copy[i][j] = player(board)

    return board_copy

def winner(board):

    def check_win(board, player):

        # Check rows
        for row in board:
            if all(cell == player for cell in row):
                return True
        # Check columns
        for col in range(len(board[0])):
            if all(board[row][col] == player for row in range(len(board))):
                return True
        # Check main diagonal
        if all(board[i][i] == player for i in range(len(board))):
            return True
        # Check secondary diagonal
        if all(board[i][len(board) - i - 1] == player for i in range(len(board))):
            return True

        return False
    """
    Returns the winner of the game, if there is one.
    """
    if check_win(board, X):
        return X
    elif check_win(board, O):
        return O
    else:
        return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # Game is over if it is a winning board or all tiles are full (no actions):

    if winner(board) == X:
        return True
    if winner(board) == O:
        return True

    for row in range(len(board)):
        for col in range(len(board[row])):
            if board[row][col] == EMPTY:
                return False
    return True

def utility(board):
    """
    Returns the utility value of the current game state.

    Returns:
    - 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    winner_player = winner(board)

    if winner_player == X:
        return 1
    elif winner_player == O:
        return -1
    else:
        return 0

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    def max_value(board):
        if terminal(board):
            return utility(board)

        v = -math.inf
        for action in actions(board):
            v = max(v, min_value(result(board, action)))
        return v

    def min_value(board):
        if terminal(board):
            return utility(board)

        v = math.inf
        for action in actions(board):
            v = min(v, max_value(result(board, action)))
        return v

    if terminal(board):
        return None

    current_player = player(board)
    plays = []

    for action in actions(board):
        if current_player == X:
            plays.append((min_value(result(board, action)), action))
        else:
            plays.append((max_value(result(board, action)), action))

    best_move = max(plays, key=lambda x: x[0] if current_player == X else -x[0])
    return best_move[1]
