import numpy as np
from Main.constants import *


class Board:
    def __init__(self):
        # Each square is 0 (empty), 1 (player1 / x), or 2 (player2 / circle)
        self.squares = np.zeros((ROWS, COLS))
        self.marked_sqrs = 0

    def final_state(self, show=False):
        """
        Returns:
            0 if there is no winner yet
            1 if player 1 (x) wins
            2 if player 2 (circle) wins
        """
        # vertical wins
        for col in range(COLS):
            if self.squares[0][col] == self.squares[1][col] == self.squares[2][col] != 0:
                if show:
                    color = CIRCLE_COLOR if self.squares[0][col] == 2 else X_COLOR
                    pygame.draw.line(screen, color, (col * SQSIZE + SQSIZE // 2, 20), (col * SQSIZE + SQSIZE // 2, HEIGHT - 20), LINE_WIDTH)
                return self.squares[0][col]

        # horizontal wins
        for row in range(ROWS):
            if self.squares[row][0] == self.squares[row][1] == self.squares[row][2] != 0:
                if show:
                    color = CIRCLE_COLOR if self.squares[row][0] == 2 else X_COLOR
                    pygame.draw.line(screen, color, (20, row * SQSIZE + SQSIZE // 2), (WIDTH - 20, row * SQSIZE + SQSIZE // 2), LINE_WIDTH)
                return self.squares[row][0]

        # descending diagonal
        if self.squares[0][0] == self.squares[1][1] ==self.squares[2][2] != 0:
            if show:
                color = CIRCLE_COLOR if self.squares[1][1] == 2 else X_COLOR
                pygame.draw.line(screen, color, (20, 20), (WIDTH - 20, HEIGHT - 20), X_WIDTH)
            return self.squares[1][1]

        # ascending diagonal
        if self.squares[2][0] == self.squares[1][1] ==self.squares[0][2] != 0:
            if show:
                color = CIRCLE_COLOR if self.squares[1][1] == 2 else X_COLOR
                pygame.draw.line(screen, color, (20, HEIGHT - 20), (WIDTH - 20, 20), X_WIDTH)
            return self.squares[1][1]

        # no winner yet
        return 0

    def mark_sqr(self, row, col, player):
        self.squares[row][col] = player
        self.marked_sqrs += 1

    def empty_sqr(self, row, col):
        return self.squares[row][col] == 0

    def get_empty_sqrs(self):
        empties = []
        for row in range(ROWS):
            for col in range(COLS):
                if self.empty_sqr(row, col):
                    empties.append((row, col))
        return empties

    def isfull(self):
        return self.marked_sqrs == 9

