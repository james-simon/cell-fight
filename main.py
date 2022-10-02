from simulator import *
import pygame


# set number of people playing
N_PLAYERS = 2
N_CPUS = 0
N_CELLS = N_PLAYERS + N_CPUS

# set size and properties of grid
LX, LY = 100, 100
px_w = 5
edge_w = 0
wrap = True

# set cell start pattern (either 'circle' or 'random')
start_pattern = 'circle'

# set whether bullets wrap
wrap_bullets = True

target_vols = [300] * (N_PLAYERS + N_CPUS)

# keys for [up, down, left, right, fire]
player_keys = {
    1: [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE],
    2: [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d, pygame.K_LSHIFT],
    3: [pygame.K_y, pygame.K_h, pygame.K_g, pygame.K_j, pygame.K_t],
    4: [pygame.K_p, pygame.K_SEMICOLON, pygame.K_l, pygame.K_QUOTE, pygame.K_o]
}

# this is packaged into a method so it can be later called to restart
def create_game():
    state = State(LX, LY, N_CELLS, colors=None, target_vols=target_vols, start_pattern=start_pattern, wrap=wrap,
                  wrap_bullets=wrap_bullets)

    for i in range(1, N_PLAYERS+1):
        state.add_player(i, player_keys[i])

    return state


state = create_game()

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 0)
pygame.init()
surface = pygame.display.set_mode((LX * px_w, LY * px_w))
clock = pygame.time.Clock()


while True:

    pygame.event.get()
    keys = pygame.key.get_pressed()
    if keys[pygame.K_p]:
        continue

    # press ESC to restart
    if keys[pygame.K_ESCAPE]:
        state = create_game()

    for _ in range(1000):
        cell_MC_step(state)
    state.step_bullets()

    draw_grid(surface, state, px_w=px_w, edge_w=edge_w)

