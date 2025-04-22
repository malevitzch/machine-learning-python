# Turns are coded in 1/8ths of a "real" turn

from enum import Enum
import pygame
import os

# this is a personal workaround so that
# the window does not appear between two screens
screen_width, screen_height = 1920, 1080
pos_x = screen_width + (screen_width - 7 * 64) // 2
pos_y = (screen_height - 7 * 64) // 2
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{pos_x},{pos_y}"


class Tile(Enum):
    NOTHING = 0
    PLAYER = 1
    ENEMY = 2


class Player:
    # moment is from 0 to 7
    def make_move(self, game):
        pass


class ManualPlayer(Player):
    def make_move(self, game):
        key = wait_for_key()
        if key in (pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d):
            nx = game.px
            ny = game.py

            if key == pygame.K_w:
                ny -= 1
            if key == pygame.K_s:
                ny += 1
            if key == pygame.K_a:
                nx -= 1
            if key == pygame.K_d:
                nx += 1
            if nx in range(7) and ny in range(7):
                if game.grid[nx][ny] == Tile.NOTHING:
                    game.grid[game.px][game.py] = Tile.NOTHING
                    game.grid[nx][ny] = Tile.PLAYER
                    game.px = nx
                    game.py = ny
                if game.grid[nx][ny] == Tile.ENEMY:
                    dx = nx - game.px
                    dy = ny - game.py
                    nex = nx + dx
                    ney = ny + dy
                    if nex in range(7) and ney in range(7):
                        if game.grid[nex][ney] == Tile.NOTHING:
                            game.grid[nx][ny] = Tile.NOTHING
                            game.grid[nex][ney] = Tile.ENEMY
                game.playercd = 4
            else:
                game.playercd = 1
        else:
            game.playercd = 1


class Game:
    def __init__(self, player):
        self.px = 3
        self.py = 3
        self.moment = 0
        self.playercd = 1
        self.player_health = 10
        self.player = player
        self.grid = [[Tile.NOTHING for _ in range(7)] for _ in range(7)]
        self.grid[3][3] = Tile.PLAYER
        self.grid[0][0] = Tile.ENEMY
        self.grid[0][6] = Tile.ENEMY
        self.grid[6][0] = Tile.ENEMY
        self.grid[6][6] = Tile.ENEMY

    def damage_player(self, damage):
        self.player_health = max(0, self.player_health - damage)

    def enemy_move(self, x, y):
        tx = x
        ty = y
        if x < self.px and self.grid[x+1][y] in (Tile.PLAYER, Tile.NOTHING):
            tx += 1
        elif x > self.px and self.grid[x-1][y] in (Tile.PLAYER, Tile.NOTHING):
            tx -= 1
        elif y < self.py and self.grid[x][y+1] in (Tile.PLAYER, Tile.NOTHING):
            ty += 1
        elif y > self.py and self.grid[x][y-1] in (Tile.PLAYER, Tile.NOTHING):
            ty -= 1
        target = self.grid[tx][ty]
        if target == Tile.NOTHING:
            self.grid[tx][ty] = Tile.ENEMY
            self.grid[x][y] = Tile.NOTHING
        elif target == Tile.PLAYER:
            self.damage_player(1)

    def step(self):
        self.playercd -= 1
        if self.playercd == 0:
            player.make_move(self)
        if self.moment != 0:
            return
        coords = []
        for i in range(7):
            for j in range(7):
                if self.grid[i][j] == Tile.ENEMY:
                    coords.append((i, j))
        for x, y in coords:
            self.enemy_move(x, y)


def wait_for_key():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.KEYDOWN:
                return event.key


player = ManualPlayer()
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
    # time.sleep(0.3)
    game.step()
    game.moment = (game.moment + 1) % 8
    if game.player_health == 0:
        running = False
pygame.quit()
