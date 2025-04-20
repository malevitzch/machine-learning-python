# Turns are coded in 1/8ths of a "real" turn

from enum import Enum
import pygame


class Tile(Enum):
    NOTHING = 0
    PLAYER = 1
    ENEMY = 2


class Player:
    # moment is from 0 to 7
    def makeMove(board, x, y, moment):
        return


class Game:
    def __init__(self, player):
        self.player = player
        self.grid = [[Tile.NOTHING for _ in range(7)] for _ in range(7)]
        self.grid[3][3] = Tile.PLAYER
        self.grid[0][0] = Tile.ENEMY
        self.grid[0][6] = Tile.ENEMY
        self.grid[6][0] = Tile.ENEMY
        self.grid[6][6] = Tile.ENEMY


player = 0
game = Game(player)

screen = pygame.display.set_mode((7 * 64, 7 * 64))
pygame.display.set_caption("The Glorious Display")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    for i in range(7):
        for j in range(7):
            width = 64
            height = 64
            if game.grid[i][j] == Tile.NOTHING:
                color = (255, 255, 255)
            elif game.grid[i][j] == Tile.ENEMY:
                color = (255, 0, 0)
            else:
                color = (0, 0, 0)
            pygame.draw.rect(screen, color,
                             (i * width, j * height,
                                 (i + 1) * width, (j + 1) * height))
    pygame.display.flip()
pygame.quit()
