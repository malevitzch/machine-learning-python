# Turns are coded in 1/8ths of a "real" turn

from enum import Enum
import pygame


class Tile(Enum):
    NOTHING = 0
    PLAYER = 1
    ENEMY = 2


class Game:
    def __init__(self, player):
        self.player = player
        self.grid = [[Tile.NOTHING for _ in range(7)] for _ in range(7)]
        self.grid[3][3] = Tile.PLAYER


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
            else:
                color = (0, 0, 0)
            pygame.draw.rect(screen, color,
                             (i * width, j * height,
                                 (i + 1) * width, (j + 1) * height))
    pygame.display.flip()
pygame.quit()
