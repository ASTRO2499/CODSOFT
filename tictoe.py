import math


# Function to print the board
def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)


# Function to check for a win or a tie
def check_winner(board):
    # Check rows
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != ' ':
            return row[0]
    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != ' ':
            return board[0][col]
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != ' ':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != ' ':
        return board[0][2]
    # Check for tie
    for row in board:
        if ' ' in row:
            return None
    return 'Tie'


# Function to get the available moves
def get_available_moves(board):
    moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                moves.append((i, j))
    return moves


# Minimax algorithm with Alpha-Beta Pruning
def minimax(board, depth, is_maximizing, alpha, beta):
    winner = check_winner(board)
    if winner == 'X':
        return -1
    elif winner == 'O':
        return 1
    elif winner == 'Tie':
        return 0

    if is_maximizing:
        max_eval = -math.inf
        for move in get_available_moves(board):
            board[move[0]][move[1]] = 'O'
            eval = minimax(board, depth + 1, False, alpha, beta)
            board[move[0]][move[1]] = ' '
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for move in get_available_moves(board):
            board[move[0]][move[1]] = 'X'
            eval = minimax(board, depth + 1, True, alpha, beta)
            board[move[0]][move[1]] = ' '
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


# Function for the AI's move
def ai_move(board):
    best_score = -math.inf
    best_move = None
    for move in get_available_moves(board):
        board[move[0]][move[1]] = 'O'
        score = minimax(board, 0, False, -math.inf, math.inf)
        board[move[0]][move[1]] = ' '
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


# Main function to play the game
def play_game():
    board = [[' ' for _ in range(3)] for _ in range(3)]
    print_board(board)

    while True:
        # Human player move
        while True:
            move = input("Enter your move (row and column): ").split()
            if len(move) != 2:
                print("Invalid input. Please enter row and column.")
                continue
            row, col = int(move[0]), int(move[1])
            if 0 <= row < 3 and 0 <= col < 3 and board[row][col] == ' ':
                board[row][col] = 'X'
                break
            else:
                print("Invalid move. Try again.")

        print_board(board)
        if check_winner(board):
            break

        # AI move
        move = ai_move(board)
        if move:
            board[move[0]][move[1]] = 'O'
        print_board(board)
        if check_winner(board):
            break

    winner = check_winner(board)
    if winner == 'Tie':
        print("It's a tie!")
    else:
        print(f"{winner} wins!")


# Start the game
if __name__ == "__main__":
    play_game()
