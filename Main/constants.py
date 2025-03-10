import pygame

# ---------- GAME CONSTANTS ----------
WIDTH = 600
HEIGHT = 600
ROWS = 3
COLS = 3
SQSIZE = WIDTH // COLS
LINE_WIDTH = 15
CIRCLE_WIDTH = 15
X_WIDTH = 20
RADIUS = SQSIZE // 4

# ---------- COLORS ----------
BG_COLOR = (12, 123, 234)
LINE_COLOR = (43, 120, 210)
CIRCLE_COLOR = (239, 231, 200)
X_COLOR = (66, 66, 66)

# ---------- PYGAME SETUP ----------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('UNBEATABLE AI')
screen.fill(BG_COLOR)
