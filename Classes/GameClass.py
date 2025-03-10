from .BoardClass import Board
from .AI import AI
from Main.constants import *


class Game:
    def __init__(self):
        self.board = Board()
        # Instead of AI(level=1, player=2),
        # let’s assume we want Q-learning for X=1
        self.ai = AI(level=2, player=2)
        self.ai.load_Q('qtable.pkl')

        self.player = 1  # 1 -> X, matches AI
        self.gamemode = 'ai'
        self.running = True
        self.show_lines()

    # ---------- DRAW METHODS ----------
    def show_lines(self):
        screen.fill(BG_COLOR)

        # vertical lines
        pygame.draw.line(screen, LINE_COLOR, (SQSIZE, 0), (SQSIZE, SQSIZE*ROWS), LINE_WIDTH)
        pygame.draw.line(screen, LINE_COLOR, (SQSIZE*2, 0), (SQSIZE*2, SQSIZE*ROWS), LINE_WIDTH)

        # horizontal lines
        pygame.draw.line(screen, LINE_COLOR, (0, SQSIZE), (SQSIZE*COLS, SQSIZE), LINE_WIDTH)
        pygame.draw.line(screen, LINE_COLOR, (0, SQSIZE*2), (SQSIZE*COLS, SQSIZE*2), LINE_WIDTH)

    def draw_fig(self, row, col):
        if self.player == 1:
            # draw x
            center = (col * SQSIZE + SQSIZE // 2, row * SQSIZE + SQSIZE // 2)
            #pygame.draw.line(screen, X_COLOR, (col * SQSIZE, row * SQSIZE), (col * SQSIZE + SQSIZE, row * SQSIZE + SQSIZE), X_WIDTH)
            #pygame.draw.line(screen, X_COLOR, (col * SQSIZE, row * SQSIZE + SQSIZE), (col * SQSIZE + SQSIZE, row * SQSIZE), X_WIDTH)
            pygame.draw.line(screen, X_COLOR, (center[0] - RADIUS,center[1] - RADIUS),(center[0]+RADIUS,center[1]+RADIUS),X_WIDTH)
            pygame.draw.line(screen, X_COLOR, (center[0] + RADIUS, center[1] - RADIUS),(center[0] - RADIUS, center[1] + RADIUS), X_WIDTH)
        else:
            # draw circle
            center = (col * SQSIZE + SQSIZE // 2, row * SQSIZE + SQSIZE // 2)
            pygame.draw.circle(screen, CIRCLE_COLOR, center, RADIUS, 15)

    # ---------- GAME LOGIC METHODS ----------
    def make_move(self, row, col):
        self.board.mark_sqr(row, col, self.player)
        self.draw_fig(row, col)
        self.next_turn()

    def next_turn(self):
        self.player = 2 if self.player == 1 else 1

    def change_gamemode(self):
        if self.gamemode == 'pvp':
            self.gamemode = 'ai'
            print("Mode has been changed to AI")
        else:
            self.gamemode = 'pvp'
            print("Mode has been changed to PVP")

    def isover(self):
        # If final_state is not 0, we have a winner
        # Or if board is full, it's over
        return self.board.final_state(show=True) != 0 or self.board.isfull()

    def reset(self):
        self.__init__()  # simple re-init
