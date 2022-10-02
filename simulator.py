import colorsys
import math
import numpy as np
import os
import pygame
import random
import time

from utils import *

# render the game grid
def draw_grid(canvas, state, px_w=10, edge_w=1):
    canvas.fill((0, 0, 0))

    boundary_px_set = set(state.boundary_pixels)
    bullet_pxs = {}
    for bullet in state.bullets:
        bullet_pxs.update({px: bullet['owner'] for px in state.bullet_pxs(bullet)})

    for r, c in np.ndindex(state.grid.shape):
        col = state.colors[state.grid[r,c]]
        if (r, c) in boundary_px_set:
            col = lighten(col,.3)
        if (r, c) in bullet_pxs and bullet_pxs[(r,c)] != state.grid[r,c]:
            col = lighten(state.colors[bullet_pxs[(r,c)]], .6)

        pygame.draw.rect(canvas, col, (c*px_w, r*px_w, px_w-edge_w, px_w-edge_w))

    pygame.display.update()

# old method. not currently used. does it still work?
def ising_step(grid, beta):
    shifted_grid_sum = sum([np.roll(2*grid - 1, a, axis=b) for (a,b) in [(1,0),(-1,0),(1,1),(-1,1)]])
    local_Es = -(2*grid - 1) * shifted_grid_sum
    probs = np.exp(beta * local_Es)
    vars1, vars2 = np.random.rand(*grid.shape), np.random.rand(*grid.shape)
    flips = (vars1 < probs) & (vars2 < .1)
    return grid*(1 - flips) + (1 - grid)*flips

# do a single local cellular-potts++ physics step
def cell_MC_step(state, beta_ising=1, beta_vol=1, beta_mov=1):
    # state: game state
    # beta_ising: inv temp for Ising energy
    # beta_vol: inv temp for volume energy
    # beta_mov: inv temp for movement energy

    # choose neighboring source and target sites.
    # the Q will be whether or not to copy source val -> target val.
    xS, yS = random.choice(state.boundary_pixels)
    (xT, yT), dir = random_neighbor(xS, yS, state.LX, state.LY, wrap=state.wrap)

    source_val = state.grid[xS,yS]
    target_val = state.grid[xT,yT]

    # if source and target have the same value, doesn't matter, so just quit
    if source_val == target_val:
        return

    # compute change in H_ising: the Ising neighbor-similarity energy
    target_neighbor_cols = neighbor_vals(state.grid, xT, yT)
    d_H_ising = (target_neighbor_cols == target_val).sum() - (target_neighbor_cols == source_val).sum()

    # compute change in H_vol: the cell volume constraint term (quadratic energy, so linear force)
    d_H_vol = 0
    if source_val != 0:
        d_H_vol += 2 * (state.cell_vols[source_val] - state.target_cell_vols[source_val]) + 1
    if target_val != 0:
        d_H_vol += -2 * (state.cell_vols[target_val] - state.target_cell_vols[target_val]) + 1

    # # random test code
    # keys = pygame.key.get_pressed()
    # if keys[pygame.K_UP]:
    #     print('UP')

    # compute desired movement direction
    if source_val != 0 and state.n_cells not in state.players:
        vs_to_other_cells = [state.cell_ctrs[i] - state.cell_ctrs[source_val]
                             for i in range(1,state.n_cells+1)
                             if i != source_val and state.cell_vols[i] > 0]
        desired_v = 10 * normalize(min(vs_to_other_cells, key=norm)) if len(vs_to_other_cells) > 0 else np.array([0,0])
    else:
        desired_v = np.array([0,0])

    pygame.event.get()

    for i in state.players:
        if source_val == i:
            desired_v = state.players[i]['speed'] * get_key_vec(*state.players[i]['keys'][:4])

    # compute change in movement energy H_v.
    # the cell wants to move "downhill" with a linear slope defined by desired_v
    d_H_mov = - (dir @ desired_v)

    prob = np.exp(-(beta_ising * d_H_ising + beta_vol * d_H_vol + beta_mov * d_H_mov))

    if random.random() > prob:
        return

    # update value
    state.grid[xT,yT] = source_val

    # update cell metabolisms
    if source_val != 0 and target_val != 0:
        state.target_cell_vols[source_val] += 1
        state.target_cell_vols[target_val] -= 1.1

    # update volumes
    state.cell_vols[source_val] += 1
    state.cell_vols[target_val] -= 1

    # recompute cell centers (using wraparound coordinates)
    state.cell_loc_sums[target_val] -= np.exp(1j * 2 * np.pi * np.array([xT,yT]) / np.array([state.LX, state.LY]))
    state.cell_loc_sums[source_val] += np.exp(1j * 2 * np.pi * np.array([xT,yT]) / np.array([state.LX, state.LY]))
    for i in [source_val, target_val]:
        thetas = np.log(state.cell_loc_sums[i] / state.cell_vols[i]).imag % (2 * np.pi)
        state.cell_ctrs[i] = thetas * np.array([state.LX, state.LY]) / (2 * np.pi)

    # update list of boundary pixels
    state.update_boundary_around_px(xT, yT)

# game state class
class State:
    def __init__(self, LX, LY, n_cells, colors=None, target_vols=30**2, start_pattern='random', wrap=True, wrap_bullets=False):
        self.LX, self.LY = LX, LY
        self.n_cells = n_cells
        self.grid = np.zeros((LX,LY))
        self.wrap = wrap

        if colors is None:
            colors = {(i + 1): hsv2rgb(i / n_cells, 1, .5) for i in range(n_cells)}
            colors[0] = (0, 0, 0)
        self.colors = colors

        if isinstance(target_vols, int):
            target_vols = [target_vols for _ in range(n_cells)]
        self.target_cell_vols = {i : target_vols[i-1] for i in range(1, n_cells+1)}

        for i in range(1, n_cells+1):
            if start_pattern == 'random':
                x, y = random.randint(LX // 5, (4 * (LX-1)) // 5), random.randint(LY // 5, (4 * (LY-1)) // 5)
            elif start_pattern == 'circle':
                theta_i = 2 * np.pi * i / n_cells
                x, y = np.array([LX,LY])/2 + np.array([LX,LY])/3 * np.array([np.cos(theta_i), np.sin(theta_i)])
                x, y = round(x), round(y)
            for dir in neighbor_dirs:
                r = mod(add((x, y), dir), self.grid.shape)
                self.grid[r[0],r[1]] = i

        self.boundary_pixels = [(x,y) for (x,y) in np.ndindex(self.grid.shape) if is_cell_boundary(self.grid, x, y, self.wrap)]
        self.cell_vols = {i : (self.grid == i).sum() for i in range(0, n_cells+1)}

        self.cell_loc_sums = {}
        self.cell_ctrs = {}
        for i in range(n_cells+1):
            self.cell_loc_sums[i] = np.array([np.exp(1j * 2 * np.pi * (1 / [LX,LY][j]) * (self.grid == i).nonzero()[j]).sum() for j in (0, 1)])
            thetas = np.log(self.cell_loc_sums[i] / self.cell_vols[i]).imag % (2 * np.pi)
            self.cell_ctrs[i] = thetas * np.array([self.LX, self.LY]) / (2 * np.pi)

        self.bullets = []
        self.bullet_cost = 10
        self.bullet_speed = 1
        self.bullet_size = 3
        self.bullet_cost = 5
        self.wrap_bullets = wrap_bullets

        self.players = {}

    # called to instantiate a new player at the start of the game
    def add_player(self, i, player_keys, speed=10):
        self.players[i] = {'keys': player_keys, 'speed': speed}

    # perform a local update
    def update_boundary_around_px(self, x, y):
        for xN,yN in [(x,y)] + neighbor_coords(x, y, self.LX, self.LY):
            if is_cell_boundary(self.grid, xN, yN, self.wrap):
                if (xN, yN) not in self.boundary_pixels:
                    self.boundary_pixels += [(xN, yN)]
            else:
                if (xN, yN) in self.boundary_pixels:
                    self.boundary_pixels.remove((xN, yN))

    # instantiate a new bullet
    def add_bullet(self, x, y, v, owner, size=1):
        self.bullets += [{'pos': np.array([x, y]),
                           'v': np.array(v),
                           'owner': owner,
                           'size': size,
                           'age': 0,
                           'wraps': 0}]

    # return all pixels containing bullets
    def bullet_pxs(self, bullet):
        xB, yB, sz = *bullet['pos'], bullet['size']
        pxs = []
        for x in range(max(round(xB - (sz - 1) / 2), 0), min(round(xB + (sz - 1) / 2) + 1, self.LX), 1):
            for y in range(max(round(yB - (sz - 1) / 2), 0), min(round(yB + (sz - 1) / 2) + 1, self.LY), 1):
                pxs += [(x,y)]
        return pxs

    # timestep bullets
    def step_bullets(self):
        keys = pygame.key.get_pressed()
        for i in self.players:
            dir_keys, fire_key = self.players[i]['keys'][:4], self.players[i]['keys'][4]
            if keys[fire_key]:
                v = get_key_vec(*dir_keys)
                if norm(v) > 0 and self.target_cell_vols[i] >= 20:
                    self.add_bullet(*self.cell_ctrs[i], 2 * v, i, self.bullet_size)
                    self.target_cell_vols[i] -= self.bullet_cost

        for b in self.bullets:
            # (xB, yB), v, owner = b['pos'], b['v'], b['owner']

            for (x,y) in self.bullet_pxs(b):
                # if a bullet's occupying a cell and that cell's not the owner...
                # ...or the bullet's traveled the radius of the field, in which case the firing grace period's over
                if self.grid[x][y] != 0 and (self.grid[x][y] != b['owner'] or b['age'] >= norm([self.LX, self.LY])/3):
                    hit_cell_id = self.grid[x][y]
                    self.target_cell_vols[hit_cell_id] -= 1
                    self.cell_vols[hit_cell_id] -= 1
                    self.grid[x][y] = 0
                    self.update_boundary_around_px(x, y)

                    # update the center of mass of the hit cell
                    self.cell_loc_sums[hit_cell_id] -= np.exp(
                        1j * 2 * np.pi * np.array([x,y]) / np.array([self.LX, self.LY]))
                    thetas = np.log(self.cell_loc_sums[hit_cell_id] / self.cell_vols[hit_cell_id]).imag % (2 * np.pi)
                    self.cell_ctrs[hit_cell_id] = thetas * np.array([self.LX, self.LY]) / (2 * np.pi)
                    # state.cell_loc_sums[source_val] += np.exp(
                    #     1j * 2 * np.pi * np.array([xT, yT]) / np.array([state.LX, state.LY]))
                    # for i in [source_val, target_val]:
                    #     thetas = np.log(state.cell_loc_sums[i] / state.cell_vols[i]).imag % (2 * np.pi)
                    #     state.cell_ctrs[i] = thetas * np.array([state.LX, state.LY]) / (2 * np.pi)

            b['pos'] += b['v']
            b['age'] += 1

        # clean out old bullets...
        if not self.wrap_bullets:
            self.bullets = [b for b in self.bullets
                            if b['pos'][0] >= 0 and b['pos'][0] <= self.LX - 1
                            and b['pos'][1] >= 0 and b['pos'][1] <= self.LY - 1]
        # ...or don't if they wrap
        else:
            for b in self.bullets:
                new_pos = mod(b['pos'], self.grid.shape)
                if not np.array_equal(new_pos, b['pos']):
                    b['wraps'] += 1
                b['pos'] = new_pos
            self.bullets = [b for b in self.bullets if b['wraps'] <= 2]