import random
import numpy as np
import math


def take_input():
    # Accepts the size of the chess board

    while True:
        try:
            size = int(input('What is the size of the chessboard? n = \n'))
            if size == 1:
                print("Trivial solution, choose a board size of at least 4")
            if size <= 3:
                print("Enter a value such that size>=4")
                continue
            return size
        except ValueError:
            print("Invalid value entered. Enter again")


def create_board(size):
    # Returns an n by n board
    board = np.random.randint(1, size, size)

    return board


def get_h_cost(board):
    h = 0
    for i in range(len(board)-1):
        # Check every column we haven't already checked
        for j in range(i + 1, len(board)):
            # Queens are in the same row
            if board[i] == board[j]:
                h += 1
            # Get the difference between the current column
            # and the check column
            offset = j - i
            # To be a diagonal, the check column value has to be equal to the current column value +/- the offset
            if board[i] == board[j] - offset or board[i] == board[j] + offset:
                h += 1
    return h


def make_move_steepest_hill(board):
    neighbour = {}
    steps = 0
    for col in range(len(board)):
        neighbours = board[col]

        for row in range(len(board)):
            if board[col] == row:
                # I don't need to evaluate the current
                # position, we already know the h-value
                continue

            board_copy = list(board)
            # Move the queen to the new row
            board_copy[col] = row
            neighbour[(col, row)] = get_h_cost(board_copy)

    neighbours = []
    h_to_beat = get_h_cost(board)

    for k, v in neighbour.items():
        if v < h_to_beat:
            h_to_beat = v

    for k, v in neighbour.items():
        if v == h_to_beat:
            neighbours.append(k)

    # Pick a random best move
    if len(neighbours) > 0:
        pick = random.randint(0, len(neighbours) - 1)
        col = neighbours[pick][0]
        row = neighbours[pick][1]
        board[col] = row

    return board


def annealing(board):
    temp = len(board) ** 2
    anneal_rate = 0.95
    new_h_cost = get_h_cost(board)
    moves = 0

    while new_h_cost > 0:
        board = make_annealing_move(board, new_h_cost, temp)

        new_h_cost = get_h_cost(board)
        print("Threads:", new_h_cost)
        # Make sure temp doesn't get impossibly low
        new_temp = max(temp * anneal_rate, 0.01)
        print("Temperature: ", new_temp)
        temp = new_temp
        moves += 1
        print("Moves: ", moves)
        if moves >= 500:
            break
    print("Annealing Solution: ", board)


def make_annealing_move(board, h_to_beat, temp):
    board_copy = list(board)
    found_move = False

    while not found_move:
        board_copy = list(board)
        new_row = random.randint(0, len(board) - 1)
        new_col = random.randint(0, len(board) - 1)
        board_copy[new_col] = new_row
        new_h_cost = get_h_cost(board_copy)
        if new_h_cost < h_to_beat:
            found_move = True
        else:
            # Check how bad was the choice it made
            delta_e = h_to_beat - new_h_cost
            # Probability can never exceed 1
            accept_probability = min(1, math.exp(delta_e / temp))
            found_move = random.random() <= accept_probability

    return board_copy


def main():
    size = take_input()
    board = create_board(size)

    print("Chessboard: ", board)

    print("Heuristic cost: ", get_h_cost(board))

    print("Steepest Hill Climb Solution: ", make_move_steepest_hill(board))

    #annealing(board)


main()
