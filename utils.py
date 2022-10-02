import colorsys
import numpy as np
import pygame

neighbor_dirs = [np.array(dir) for dir in [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]]

# HSV -> RGB
def hsv2rgb(h, s, v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))

# lighten an RGB color
def lighten(col, factor=.5):
    assert factor >= 0 and factor <= 1
    return tuple([255 * factor + c * (1 - factor) for c in col])

# darken an RGB color
def darken(col, factor=.5):
    assert factor >= 0 and factor <= 1
    return tuple([c * (1 - factor) for c in col])

# return vector norm
def norm(vec):
    vec = np.array(vec)
    return (vec @ vec)**.5

# return normalized vector
def normalize(vec):
    if norm(vec) == 0:
        return vec
    return vec / norm(vec)

# add vectors
def add(v1, v2):
    assert len(v1) == len(v2)
    if type(v1) == np.ndarray and type(v2) == np.ndarray:
        return v1 + v2
    return [v1[i] + v2[i] for i in range(len(v1))]

# elementwise v % q
def mod(v, q):
    assert len(v) == len(v)
    if type(v) == np.ndarray and type(q) == np.ndarray:
        return v % q
    return [v[i] % q[i] for i in range(len(v))]

# checks that every element v[i] is in the range [0, ..., q[i]]
def vec_in_rectangle(v, q):
    return mod(v, q) == v

# arrowkey press -> [vx,vy]
def get_key_vec(upkey, downkey, leftkey, rightkey):
    vx, vy = 0, 0
    keys = pygame.key.get_pressed()
    if keys[upkey]:
        vx -= 1
    if keys[downkey]:
        vx += 1
    if keys[leftkey]:
        vy -= 1
    if keys[rightkey]:
        vy += 1
    return normalize(np.array([vx,vy]))

# checks whether (x,y) is different in value from any of its neighbors
def is_cell_boundary(grid, x, y, wrap=True):
    col = grid[x][y]
    if wrap:
        for dir in neighbor_dirs:
            if grid[(x + dir[0]) % grid.shape[0], (y + dir[1]) % grid.shape[1]] != col:
                return True
    else:
        for dir in neighbor_dirs:
            if grid[(x + dir[0]) % grid.shape[0], (y + dir[1]) % grid.shape[1]] != col:
                if (x + dir[0]) != -1 and (y + dir[1]) != -1 and (x + dir[0]) != grid.shape[0] and (y + dir[1]) != grid.shape[1]:
                    return True
    return False

# get a random neighbor cell to (x,y) mod (LX,LY)
def random_neighbor(x, y, LX, LY, wrap=True):
    while True:
        dir = neighbor_dirs[np.random.randint(0, 8)]
        xN, yN = add([x,y], dir)
        if wrap or (xN != -1 and yN != -1 and xN != LX and yN != LY):
            break
    return mod((xN, yN), (LX, LY)), dir

def neighbor_coords(x, y, LX, LY, wrap=True):
    if wrap:
        return [((x + dir[0]) % LX, (y + dir[1]) % LY) for dir in neighbor_dirs]
    else:
        return [((x + dir[0]), (y + dir[1])) for dir in neighbor_dirs
                if (x + dir[0]) != -1 and (y + dir[1]) != -1 and (x + dir[0]) != LX and (y + dir[1]) != LY]

def neighbor_vals(grid, x, y, wrap=True):
    valid_dirs = neighbor_dirs if wrap \
        else [dir for dir in neighbor_dirs if vec_in_rectangle(add((x, y), dir), grid.shape)]
    return np.array([grid[(x + dir[0]) % grid.shape[0], (y + dir[1]) % grid.shape[1]] for dir in valid_dirs])

    # if wrap:
    #     return np.array([grid[(x + dir[0]) % grid.shape[0], (y + dir[1]) % grid.shape[1]] for dir in neighbor_dirs])
    # else:
    #     return np.array([grid[(x + dir[0]) % grid.shape[0], (y + dir[1]) % grid.shape[1]]
    #                      for dir in neighbor_dirs
    #                      if (x + dir[0]) != -1 and (y + dir[1]) != -1 and (x + dir[0]) != LX and (y + dir[1]) != LY])
