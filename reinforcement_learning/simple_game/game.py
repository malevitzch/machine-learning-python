# Turns are coded in 1/8ths of a "real" turn

from enum import Enum


class Tile(Enum):
    NOTHING = 0
    PLAYER = 1
    ENEMY = 2


class Game:
    def __init__(self, player):
        self.player = player
        self.grid = [[None for _ in range(7)] for _ in range(7)]
