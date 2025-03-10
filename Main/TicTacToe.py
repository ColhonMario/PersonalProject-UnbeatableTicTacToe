import sys
from Classes.GameClass import *


def main():

    game = Game()
    board = game.board
    ai = game.ai

    while True:
        # Pygame event loop
        for event in pygame.event.get():
            # Quit event
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Keydown events
            if event.type == pygame.KEYDOWN:
                # g -> toggle gamemode
                if event.key == pygame.K_g:
                    game.change_gamemode()

                # r -> restart
                if event.key == pygame.K_r:
                    game.reset()
                    board = game.board
                    ai = game.ai

                # 0 -> random AI
                if event.key == pygame.K_0:
                    ai.level = 0
                    print("Random AI")

                # 1 -> minimax AI
                if event.key == pygame.K_1:
                    ai.level = 1
                    print("Minimax AI")

                if event.key == pygame.K_2:
                    ai.level = 2
                    print("Q-learning AI")

            # Mouse click events
            if event.type == pygame.MOUSEBUTTONDOWN:
                if game.running:
                    pos = event.pos
                    row = pos[1] // SQSIZE
                    col = pos[0] // SQSIZE

                    if board.empty_sqr(row, col):
                        game.make_move(row, col)
                        if game.isover():
                            game.running = False

        # If it's AI's turn and the game is still running
        if game.gamemode == 'ai' and game.player == ai.player and game.running:
            pygame.display.update()
            # AI evaluates and chooses a move
            row, col = ai.eval(board)
            game.make_move(row, col)
            if game.isover():
                game.running = False

        pygame.display.update()


if __name__ == "__main__":
    main()
