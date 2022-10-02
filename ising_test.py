from simulator import *
import pygame

# set size and properties of grid
LX, LY = 100, 100
px_w = 5
edge_w = 0
wrap = True

state = State(LX, LY, 1)
state.boundary_pixels = []
state.colors[1] = [255, 255, 255]
state.grid = np.random.random_integers(0, 1, state.grid.shape)

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 0)
pygame.init()
surface = pygame.display.set_mode((LX * px_w, LY * px_w))
clock = pygame.time.Clock()

while True:
    pygame.event.get()
    keys = pygame.key.get_pressed()
    if keys[pygame.K_p]:
        continue

    for _ in range(10):
        state.grid = ising_step(state.grid, 3)
    state.step_bullets()

    draw_grid(surface, state, px_w=px_w, edge_w=edge_w)

