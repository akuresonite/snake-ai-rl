"""Vectorized Snake DQN: many games at once, one network call per step.

Why this is faster than the single game script
----------------------------------------------
snake_ai_training_Deep.py advances ONE snake and calls the network once per
frame, on a batch of one sample. Measured on an RTX 5050 laptop that single
sample step costs 3713 us on the GPU against 2610 us on the CPU, because the
GPU spends all its time on launch overhead, not arithmetic.

This script advances N snakes per frame and asks the network for all N actions
in ONE call. The batch is now N wide, so the GPU finally has something to chew
on, and the per sample cost collapses. The same idea applies to training: one
replay batch covers N games worth of new experience.

What it does NOT do
-------------------
It does not make a single run converge in fewer games. More games run in
parallel means more experience per wall clock second, which in practice means
more learning per hour. The sample efficiency per game is unchanged.

Usage
-----
    python snake_ai_training_vec.py --games 64 --steps 2000000
    python snake_ai_training_vec.py --games 32 --device cpu
"""

import argparse
import os
import random
import time
from collections import deque

import numpy as np
import torch

# Shared pieces come from snake_core, which is pygame free. Importing the
# single game script instead would drag pygame in and this script would fail on
# any environment that has torch but no pygame, which is exactly the case for
# the CUDA equipped conda envs on this machine.
from snake_core import (  # noqa: E402
    BATCH_SIZE,
    BLOCK_SIZE,
    DEVICE,
    EPSILON_DECAY_GAMES,
    EPSILON_MIN,
    EPSILON_START,
    GAMMA,
    HIDDEN_LAYERS,
    HIDDEN_SIZE,
    LR,
    MAX_MEMORY,
    REWARD_DEATH,
    REWARD_FOOD,
    SNAKE_THREADS,
    STARVATION_FRAMES_PER_SEGMENT,
    WEIGHT_DECAY,
    Direction,
    Linear_QNet,
    QTrainer,
    architecture_fingerprint,
    build_model,
    grid_architecture_fingerprint,
    set_threads,
    torch_load_checkpoint,
)

from snake_logging import Logger, NullLogger, add_histogram  # noqa: E402

# This script does wide batched work, which is the case where threads can
# actually help. Unlike the single game script it does not force one thread, so
# a user on a CUDA box can tune it. Override with SNAKE_THREADS=8 if you like.
THREADS = set_threads(SNAKE_THREADS) if SNAKE_THREADS else None

VEC_CHECKPOINT = "./model/vec_checkpoint.pth"
VEC_BEST_CHECKPOINT = "./model/vec_best.pth"


# ==========================================
# RUN NAMING
# ==========================================
# TensorBoard lists runs alphabetically by folder name, so the folder name is
# what the sidebar looks like. One scheme, used by the trainer, the sweep runner
# and tools/rename_runs.py, so all three agree:
#
#     <group>__<variant>__s<seed>
#
#   group    what is compared, s01, s02 and so on, so groups sort in sweep order
#   variant  what makes the run different, letters digits dash underscore only.
#            TensorBoard splits a name on slashes, so no slashes, no spaces.
#   seed     the random seed, so repeats sit next to each other
#
# Example: s03__target-double__s0


def snake_variant_desc(obs_mode="basic", use_norm=True, target_net=False,
                       double_dqn=False, n_step=1):
    """Short description of what makes a run different from the baseline."""
    bits = []
    if obs_mode != "basic":
        bits.append(obs_mode)
    if not use_norm:
        bits.append("no-norm")
    if target_net:
        bits.append("target")
        if double_dqn:
            bits.append("double")
    if n_step and int(n_step) > 1:
        bits.append(f"n{int(n_step)}")
    return "-".join(bits) if bits else "baseline"


def snake_run_name(group, obs_mode="basic", use_norm=True, target_net=False,
                   double_dqn=False, n_step=1, seed=None):
    """Build the folder name for one run."""
    variant = snake_variant_desc(obs_mode, use_norm, target_net, double_dqn, n_step)
    name = f"{group}__{variant}"
    if seed is not None:
        name += f"__s{seed}"
    return name


# ==========================================
# BOARD GEOMETRY, no pygame
# ==========================================

class Board:
    """Just the rules. No window, no clock, no drawing.

    The single game class carries pygame calls through every method, which makes
    it impossible to run several of them in a tight loop. This class is the same
    rules with the rendering stripped out.
    """

    __slots__ = ("cols", "rows", "game_w", "game_h")

    def __init__(self, w=800, h=600, block=BLOCK_SIZE, cols=None, rows=None):
        if cols and rows:
            self.cols = int(cols)
            self.rows = int(rows)
        else:
            self.cols = w // block
            self.rows = h // block
        self.game_w = self.cols * block
        self.game_h = self.rows * block

    def cell_count(self):
        return self.cols * self.rows


# ==========================================
# PART 1: VECTORIZED ENVIRONMENT
# ==========================================

class VecSnake:
    """N independent snakes advanced together.

    State layout is one numpy array per fact, so everything can be gathered and
    written back in a single shot instead of looping over games in Python.

    obs_mode picks what the agent can see. The width is owned HERE, on purpose:
    an earlier experiment built a 19 input network but kept 11 wide samples in
    replay, so training raised and the result was noise. Keeping the width next
    to the code that fills it makes that mistake impossible.

      "basic"  11 features. The original. One cell of lookahead.
      "rich"   21 features. Wall distance, distance to the tail, length, and
               the distance to the food.
      "flood"  25 features. Everything in basic, plus the flood fill numbers.
               These are the ones about being trapped, which matters because
               95 to 100 percent of deaths are the snake running into itself.
               Measured, frames per food rises only slightly with score while
               self collisions stay at almost every death, so the agent is
               sealing itself in rather than running out of time.
      "grid"   4 channels of the whole board: body, head, food, and the cell
               straight ahead. Read by a small conv net with global pooling,
               so one set of weights reads any board size. This is the fix for
               the information wall the flat features hit, and the size
               transfer is the curriculum experiment.
    """

    WIDTHS = {"basic": 11, "rich": 21, "flood": 25}

    def __init__(self, n_games, w=800, h=600, auto_reset=True, obs_mode="basic",
                 cols=None, rows=None, step_reward=0.0, distance_reward=0.0):
        if obs_mode not in ("basic", "rich", "flood", "grid"):
            raise ValueError(f"obs_mode must be one of basic, rich, flood, grid")
        self.n = n_games
        self.obs_mode = obs_mode
        self.obs_width = self.WIDTHS.get(obs_mode)
        self.board = Board(w, h, cols=cols, rows=rows)
        self.auto_reset = auto_reset
        # A small penalty on steps that are not eating. This is the fix for the
        # circling loophole: with a 0 reward for doing nothing, Q(circle) stays 0
        # forever and the food signal never propagates, so a conv net on a big
        # board learns to circle until the starvation guard kills it. Measured:
        # 12 of 12 deaths by starvation, food about 14 cells away. A negative
        # step reward makes circling strictly worse than eating, so the value
        # gradient towards the food finally exists.
        self.step_reward = float(step_reward)
        # Distance shaping: reward for moving closer to the food, penalty for
        # moving away. The flat 11 feature state hands the DQN the food
        # direction for free, which is why it learns to eat quickly. A conv net
        # must discover the food direction itself, and on a 40x30 board the food
        # is usually outside its receptive field, so the gradient has to come
        # from the reward instead. This is the standard snake tutorial trick.
        self.distance_reward = float(distance_reward)
        self.steps = 0
        self.episodes = 0
        self.episode_returns = deque(maxlen=200)
        self.episode_scores = deque(maxlen=200)
        self.episode_lengths = deque(maxlen=200)
        self.episode_best = 0
        self._returns = np.zeros(n_games, dtype=np.float32)

        # The raw board, kept incrementally so it costs one cell write per
        # change instead of a rebuild per step. observe() returns a copy,
        # because the replay buffer must keep the state as it was and the grid
        # keeps changing under it.
        if obs_mode == "grid":
            self.grids = np.zeros((n_games, 4, self.board.rows, self.board.cols),
                                  dtype=np.uint8)
            self.ahead_cells = np.zeros((n_games, 2), dtype=np.int32)

        # Scratch space for the flood fill, allocated once. Reusing these is what
        # makes the flood state usable at all: a set based version ran the loop
        # at 71 steps per second against 19,000 for the basic state.
        cells = self.board.cols * self.board.rows
        self._occ = np.zeros(cells, dtype=bool)
        self._seen = np.zeros(cells, dtype=bool)
        self._stack = np.zeros(cells, dtype=np.int32)

        self.reset()

    # ---- helpers ----------------------------------------------------------
    def _is_collision(self, idx, x, y):
        """True if cell (x, y) is a wall or part of snake idx."""
        if x < 0 or y < 0 or x > self.board.game_w - BLOCK_SIZE or y > self.board.game_h - BLOCK_SIZE:
            return True
        return (x, y) in self.bodies[idx]

    def _place_food(self, idx):
        cols, rows = self.board.cols, self.board.rows
        body = self.bodies[idx]
        for _ in range(64):
            c = random.randrange(cols)
            r = random.randrange(rows)
            if (c * BLOCK_SIZE, r * BLOCK_SIZE) not in body:
                return c * BLOCK_SIZE, r * BLOCK_SIZE
        # Dense board, fall back to a scan.
        for c in range(cols):
            for r in range(rows):
                if (c * BLOCK_SIZE, r * BLOCK_SIZE) not in body:
                    return c * BLOCK_SIZE, r * BLOCK_SIZE
        return -1, -1  # board full

    def _reset_one(self, idx):
        cx = (self.board.cols // 2) * BLOCK_SIZE
        cy = (self.board.rows // 2) * BLOCK_SIZE
        self.heads[idx] = (cx, cy)
        self.directions[idx] = Direction.RIGHT.value
        self.bodies[idx] = {(cx, cy), (cx - BLOCK_SIZE, cy), (cx - 2 * BLOCK_SIZE, cy)}
        self.queues[idx] = deque([(cx, cy), (cx - BLOCK_SIZE, cy), (cx - 2 * BLOCK_SIZE, cy)])
        self.scores[idx] = 0
        self.frames[idx] = 0
        self._returns[idx] = 0.0
        fx, fy = self._place_food(idx)
        self.foods[idx] = (fx, fy)
        if self.obs_mode == "grid":
            self._build_grid(idx)

    def _build_grid(self, idx):
        """Rebuild the 4 channel board for one game. Called on reset only, the
        step loop updates cells in place."""
        g = self.grids[idx]
        g.fill(0)
        cols, rows = self.board.cols, self.board.rows
        for x, y in self.bodies[idx]:
            gx, gy = x // BLOCK_SIZE, y // BLOCK_SIZE
            if 0 <= gx < cols and 0 <= gy < rows:
                g[0, gy, gx] = 1
        hx, hy = int(self.heads[idx, 0]) // BLOCK_SIZE, int(self.heads[idx, 1]) // BLOCK_SIZE
        g[1, hy, hx] = 1
        fx, fy = int(self.foods[idx, 0]) // BLOCK_SIZE, int(self.foods[idx, 1]) // BLOCK_SIZE
        if 0 <= fx < cols and 0 <= fy < rows:
            g[2, fy, fx] = 1
        ax, ay = self._straight_cell(idx)
        if 0 <= ax < cols and 0 <= ay < rows:
            g[3, ay, ax] = 1
        self.ahead_cells[idx] = (ax, ay)

    def _straight_cell(self, idx):
        """The cell the snake enters on action straight. Channel 3 of the grid.

        This channel is what makes the three RELATIVE actions readable from an
        absolute board: the conv net sees which cell is ahead, and learns that
        action 0 goes there, action 1 right of the heading, action 2 left.
        """
        hx, hy = int(self.heads[idx, 0]), int(self.heads[idx, 1])
        d = self.directions[idx]
        if d == Direction.RIGHT.value:
            return hx // BLOCK_SIZE + 1, hy // BLOCK_SIZE
        if d == Direction.LEFT.value:
            return hx // BLOCK_SIZE - 1, hy // BLOCK_SIZE
        if d == Direction.DOWN.value:
            return hx // BLOCK_SIZE, hy // BLOCK_SIZE + 1
        return hx // BLOCK_SIZE, hy // BLOCK_SIZE - 1

    def reset(self):
        self.heads = np.zeros((self.n, 2), dtype=np.int32)
        self.directions = np.zeros(self.n, dtype=np.int32)
        self.scores = np.zeros(self.n, dtype=np.int32)
        self.frames = np.zeros(self.n, dtype=np.int32)
        self.foods = np.zeros((self.n, 2), dtype=np.int32)
        self.bodies = [None] * self.n
        self.queues = [None] * self.n
        for i in range(self.n):
            self._reset_one(i)
        return self.observe()

    # ---- observation ------------------------------------------------------
    def observe(self):
        """The state for every game, shaped by obs_mode.

        Flat modes return (n, obs_width) int8. Grid mode returns
        (n, 4, rows, cols) uint8 and the copy matters: the grids are updated in
        place by step(), so handing out the live array would corrupt every
        sample the replay buffer is holding.
        """
        n = self.n
        if self.obs_mode == "grid":
            return self.grids.copy()
        if self.obs_mode == "basic":
            return self._observe_basic(n)
        if self.obs_mode == "rich":
            return self._observe_rich(n)
        return self._observe_flood(n)

    # ---- flood fill, the trap measurement --------------------------------
    #
    # Performance matters here. A set of tuples costs roughly 12 microseconds
    # per call, and trap_stats calls it up to four times per game per step, so
    # with 256 games that is milliseconds per environment step. Measured, the
    # set version ran the loop at 71 steps per second against 19,000 for the
    # basic state, which makes it unusable.
    #
    # This version uses flat numpy arrays and a preallocated stack. No sets, no
    # tuples, no allocation per call.
    def _fill_occupancy(self, body, head, tail):
        """Mark body cells, minus the head and the tail, as blocked."""
        self._occ.fill(False)
        for x, y in body:
            self._occ[(y // BLOCK_SIZE) * self.board.cols + (x // BLOCK_SIZE)] = True
        # The head is where the snake already is, not a wall. Leaving it blocked
        # made the area from the head come out as 0, which turned every trap
        # feature into a constant.
        self._occ[(head[1] // BLOCK_SIZE) * self.board.cols + (head[0] // BLOCK_SIZE)] = False
        if tail is not None:
            self._occ[(tail[1] // BLOCK_SIZE) * self.board.cols + (tail[0] // BLOCK_SIZE)] = False

    def flood_area_grid(self, start_gx, start_gy):
        """Count reachable cells from a grid cell, using the occupancy grid."""
        cols, rows = self.board.cols, self.board.rows
        if start_gx < 0 or start_gy < 0 or start_gx >= cols or start_gy >= rows:
            return 0
        start = start_gy * cols + start_gx
        if self._occ[start]:
            return 0
        seen = self._seen
        seen.fill(False)
        seen[start] = True
        stack = self._stack
        top = 0
        stack[0] = start
        top = 1
        area = 0
        while top:
            top -= 1
            cur = stack[top]
            area += 1
            cx = cur % cols
            cy = cur // cols
            if cx > 0:
                n = cur - 1
                if not seen[n] and not self._occ[n]:
                    seen[n] = True
                    stack[top] = n
                    top += 1
            if cx < cols - 1:
                n = cur + 1
                if not seen[n] and not self._occ[n]:
                    seen[n] = True
                    stack[top] = n
                    top += 1
            if cy > 0:
                n = cur - cols
                if not seen[n] and not self._occ[n]:
                    seen[n] = True
                    stack[top] = n
                    top += 1
            if cy < rows - 1:
                n = cur + cols
                if not seen[n] and not self._occ[n]:
                    seen[n] = True
                    stack[top] = n
                    top += 1
        return area

    def _tail_reachable_grid(self, tail_g):
        return self._seen[tail_g[1] * self.board.cols + tail_g[0]]

    def trap_stats(self, idx):
        """Everything needed to describe whether a snake is trapping itself."""
        queues = self.queues[idx]
        body = self.bodies[idx]
        hx, hy = int(self.heads[idx, 0]), int(self.heads[idx, 1])
        head = (hx, hy)
        tail = queues[-1] if len(queues) > 1 else None
        self._fill_occupancy(body, head, tail)

        cells = self.board.cols * self.board.rows
        gx, gy = hx // BLOCK_SIZE, hy // BLOCK_SIZE
        area_from_head = self.flood_area_grid(gx, gy)
        # The reachable set from the head is now in self._seen, so the tail check
        # is free rather than a second search.
        tail_reachable = True
        if tail is not None:
            tail_reachable = self._tail_reachable_grid(
                (tail[0] // BLOCK_SIZE, tail[1] // BLOCK_SIZE))

        danger, _named = self.collision_detail(idx)
        legal = [m for m in ("straight", "right", "left") if not danger[m]]
        best_move_area = 0
        per_move = {}
        for m in legal:
            cx, cy = self.ray_cell(head, self.directions[idx], m)
            # flood_area_grid overwrites self._seen, so the tail check above has
            # to happen first. It does.
            per_move[m] = self.flood_area_grid(cx // BLOCK_SIZE, cy // BLOCK_SIZE)
            best_move_area = max(best_move_area, per_move[m])

        body_cells = max(1, len(body) - (1 if tail is not None else 0) - 1)
        free_cells = cells - body_cells
        return {
            "reachable_from_head": area_from_head,
            "best_move_area": best_move_area,
            "per_move_area": per_move,
            "legal_moves": len(legal),
            "trap_ratio": area_from_head / cells,
            "free_ratio": free_cells / cells,
            "tail_reachable": tail_reachable,
            "cells": cells,
            "body": body_cells,
            "free": free_cells,
        }

    def collision_detail(self, idx):
        """Danger per move name, and the cell each move would enter."""
        head = (int(self.heads[idx, 0]), int(self.heads[idx, 1]))
        d = self.directions[idx]
        clock = (Direction.RIGHT.value, Direction.DOWN.value,
                 Direction.LEFT.value, Direction.UP.value)
        i = clock.index(d)
        named = {
            "straight": clock[i],
            "right": clock[(i + 1) % 4],
            "left": clock[(i - 1) % 4],
        }
        out = {}
        for name, nd in named.items():
            cell = self.ray_cell(head, d, name)
            out[name] = self._is_collision(idx, *cell)
        return out, named

    @staticmethod
    def ray_cell(head, d, move):
        clock = (Direction.RIGHT.value, Direction.DOWN.value,
                 Direction.LEFT.value, Direction.UP.value)
        i = clock.index(d)
        if move == "straight":
            nd = clock[i]
        elif move == "right":
            nd = clock[(i + 1) % 4]
        else:
            nd = clock[(i - 1) % 4]
        x, y = head
        if nd == Direction.RIGHT.value:
            x += BLOCK_SIZE
        elif nd == Direction.LEFT.value:
            x -= BLOCK_SIZE
        elif nd == Direction.DOWN.value:
            y += BLOCK_SIZE
        else:
            y -= BLOCK_SIZE
        return (x, y)

    def _observe_flood(self, n):
        """Basic 11, then the flood fill numbers scaled to 0..9.

        Cost note: trap_stats calls flood fill up to four times per game, so
        this is about 40 observations per second for a single game and about
        152 steps per second for a training loop, against 94,610 for the basic
        state. It is usable for recording a teacher or for analysis with a
        handful of games, and NOT usable for a training sweep at full stride.
        Use --obs-stride in tools/ml_imitation.py to sample instead.
        """
        out = np.zeros((n, 25), dtype=np.int8)
        out[:, :11] = self._observe_basic(n)
        for i in range(n):
            st = self.trap_stats(i)
            cells = st["cells"]
            # Reachable area as a fraction of the board, in tenths.
            out[i, 11] = min(9, int(st["trap_ratio"] * 10))
            out[i, 12] = min(9, int(st["best_move_area"] / cells * 10))
            out[i, 13] = st["legal_moves"]
            out[i, 14] = 1 if st["tail_reachable"] else 0
            for j, m in enumerate(("straight", "right", "left")):
                if m in st["per_move_area"]:
                    out[i, 15 + j] = min(9, int(st["per_move_area"][m] / cells * 10))
                else:
                    out[i, 15 + j] = -1
            out[i, 18] = min(9, int(st["free_ratio"] * 10))
            out[i, 19] = min(9, int(st["body"] / cells * 10))
            out[i, 20] = 9 - out[i, 11]
            out[i, 21] = 1 if st["best_move_area"] < st["body"] else 0
            out[i, 22] = min(9, int(st["reachable_from_head"] / max(1, st["free"]) * 9))
            out[i, 23] = 9 - min(9, int(st["free_ratio"] * 10))
            out[i, 24] = 1 if st["legal_moves"] <= 1 else 0
        return out

    def observe_flood_fast(self):
        """Deprecated. Slower than observe(), kept only as a warning.

        Measured 36 observations per second against 75 for observe(), because it
        calls observe() internally and that already runs trap_stats. The lesson:
        the cost is the flood fill itself, not the array handling around it, so
        micro optimising the wrapper cannot help. Use observation striding
        instead, which cuts how OFTEN the flood runs rather than how fast it is.
        """
        return self.observe()

    def _observe_basic(self, n):
        out = np.zeros((n, 11), dtype=np.int8)
        for i in range(n):
            hx, hy = int(self.heads[i, 0]), int(self.heads[i, 1])
            d = self.directions[i]
            straight, r_turn, l_turn = self._rays(hx, hy, d)

            out[i, 0] = self._is_collision(i, *straight)
            out[i, 1] = self._is_collision(i, *r_turn)
            out[i, 2] = self._is_collision(i, *l_turn)

            out[i, 3] = d == Direction.LEFT.value
            out[i, 4] = d == Direction.RIGHT.value
            out[i, 5] = d == Direction.UP.value
            out[i, 6] = d == Direction.DOWN.value

            fx, fy = int(self.foods[i, 0]), int(self.foods[i, 1])
            if fx < 0:  # board full, no food
                continue
            out[i, 7] = fx < hx
            out[i, 8] = fx > hx
            out[i, 9] = fy < hy
            out[i, 10] = fy > hy
        return out

    def _observe_rich(self, n):
        out = np.zeros((n, 21), dtype=np.int8)
        bw = self.board.game_w - BLOCK_SIZE
        bh = self.board.game_h - BLOCK_SIZE
        cols = max(1, self.board.cols - 1)
        # Divide distances by 10 so they land in 0..3. The furthest wall on an
        # 800 by 600 board is 39 cells away, so this keeps them inside a range
        # the other 0 or 1 features can compete with.
        scale = 10
        for i in range(n):
            hx, hy = int(self.heads[i, 0]), int(self.heads[i, 1])
            d = self.directions[i]
            straight, r_turn, l_turn = self._rays(hx, hy, d)

            out[i, 0] = self._is_collision(i, *straight)
            out[i, 1] = self._is_collision(i, *r_turn)
            out[i, 2] = self._is_collision(i, *l_turn)

            # Distance to each wall, in scaled cells.
            out[i, 3] = (hx // BLOCK_SIZE) // scale
            out[i, 4] = ((bw - hx) // BLOCK_SIZE) // scale
            out[i, 5] = (hy // BLOCK_SIZE) // scale
            out[i, 6] = ((bh - hy) // BLOCK_SIZE) // scale

            out[i, 7] = d == Direction.LEFT.value
            out[i, 8] = d == Direction.RIGHT.value
            out[i, 9] = d == Direction.UP.value
            out[i, 10] = d == Direction.DOWN.value

            # Which way the tail is. Tail direction is cheap and it varies.
            q = self.queues[i]
            if len(q) >= 2:
                tx, ty = q[-1]
                out[i, 11] = tx < hx
                out[i, 12] = tx > hx
                out[i, 13] = ty < hy
                out[i, 14] = ty > hy
            # Snake length, log scaled to 1..9.
            #
            # Honest note: this feature is nearly useless in practice. Measured
            # over 256,000 observations, raw length during random play sits
            # between 3 and 5, because a random snake dies almost immediately.
            # So this is a constant 1 for the whole early game, which is exactly
            # when a length signal would be worth having. Kept because it costs
            # nothing once the agent improves and games get longer.
            out[i, 15] = min(9, int(np.log1p(len(q))))

            fx, fy = int(self.foods[i, 0]), int(self.foods[i, 1])
            if fx < 0:
                pass
            else:
                out[i, 16] = fx < hx
                out[i, 17] = fx > hx
                out[i, 18] = fy < hy
                out[i, 19] = fy > hy
                # How far the food is, log scaled. The four bits above say which
                # way it is, this says how far, which matters for route planning.
                manhattan = abs(fx - hx) + abs(fy - hy)
                out[i, 20] = min(9, int(np.log1p(manhattan // BLOCK_SIZE)))
        return out

    @staticmethod
    def _rays(hx, hy, d):
        """Straight, right turn and left turn cells for a head facing d."""
        left = (hx - BLOCK_SIZE, hy)
        right = (hx + BLOCK_SIZE, hy)
        up = (hx, hy - BLOCK_SIZE)
        down = (hx, hy + BLOCK_SIZE)
        if d == Direction.RIGHT.value:
            return right, down, up
        if d == Direction.DOWN.value:
            return down, left, up
        if d == Direction.LEFT.value:
            return left, up, down
        return up, right, left

    # ---- stepping ---------------------------------------------------------
    def step(self, actions):
        """actions: (n,) ints 0=straight 1=right 2=left.

        Returns obs, rewards, done, scores. done is the true terminal flag for
        the transition that just happened. If auto_reset is on, the games are
        already restarted by the time you read obs.
        """
        n = self.n
        self.steps += n
        self.frames += 1

        rewards = np.zeros(n, dtype=np.float32)
        done = np.zeros(n, dtype=bool)
        scores = self.scores.copy()

        for i in range(n):
            old_hx, old_hy = int(self.heads[i, 0]), int(self.heads[i, 1])
            hx, hy = old_hx, old_hy
            d = self.directions[i]
            a = int(actions[i])

            # RIGHT=1, LEFT=2, UP=3, DOWN=4 laid out clockwise, same as the
            # single game version.
            clock = (Direction.RIGHT.value, Direction.DOWN.value, Direction.LEFT.value, Direction.UP.value)
            idx = clock.index(d)
            if a == 0:
                d = clock[idx]
            elif a == 1:
                d = clock[(idx + 1) % 4]
            else:
                d = clock[(idx - 1) % 4]
            self.directions[i] = d

            if d == Direction.RIGHT.value:
                hx += BLOCK_SIZE
            elif d == Direction.LEFT.value:
                hx -= BLOCK_SIZE
            elif d == Direction.DOWN.value:
                hy += BLOCK_SIZE
            elif d == Direction.UP.value:
                hy -= BLOCK_SIZE

            body = self.bodies[i]
            fx, fy = int(self.foods[i, 0]), int(self.foods[i, 1])

            # A full board means the snake won, there is nothing left to eat.
            if fx < 0:
                done[i] = True
                rewards[i] = REWARD_FOOD
                self._finish(i, scores[i], rewards[i])
                continue

            alive = not self._is_collision(i, hx, hy)
            if alive:
                alive = self.frames[i] <= STARVATION_FRAMES_PER_SEGMENT * len(body)

            if not alive:
                done[i] = True
                rewards[i] = REWARD_DEATH
                self._finish(i, scores[i], rewards[i])
                continue

            # Move: new head in, tail out unless food was eaten.
            new_cell = (hx, hy)
            self.queues[i].appendleft(new_cell)
            body.add(new_cell)
            self.heads[i] = (hx, hy)

            if new_cell == (fx, fy):
                self.scores[i] += 1
                rewards[i] = REWARD_FOOD
                self.frames[i] = 0
                nx, ny = self._place_food(i)
                self.foods[i] = (nx, ny)
            else:
                tail = self.queues[i].pop()
                body.discard(tail)
                # The neutral step penalty. See step_reward on __init__.
                rewards[i] += self.step_reward
                # Distance shaping towards the food, only on non eating steps.
                # The eaten cell moved the food, so shaping against the old food
                # position would be noise. In cells, so the scale is the same on
                # every board size.
                if self.distance_reward:
                    old_d = abs(fx - old_hx) + abs(fy - old_hy)
                    new_d = abs(fx - hx) + abs(fy - hy)
                    rewards[i] += (old_d - new_d) // BLOCK_SIZE * self.distance_reward
                if self.obs_mode == "grid":
                    # The tail cell is empty again, one write, no rebuild.
                    tx, ty = tail[0] // BLOCK_SIZE, tail[1] // BLOCK_SIZE
                    if 0 <= tx < self.board.cols and 0 <= ty < self.board.rows:
                        self.grids[i, 0, ty, tx] = 0

            if self.obs_mode == "grid":
                self._grid_step(i, old_hx, old_hy, fx, fy)

        self._returns += rewards
        if self.auto_reset:
            for i in range(n):
                if done[i]:
                    self._reset_one(i)

        return self.observe(), rewards, done, scores

    def _finish(self, idx, score, reward):
        self.episodes += 1
        self.episode_returns.append(float(self._returns[idx]))
        self.episode_scores.append(int(score))
        self.episode_lengths.append(int(self.frames[idx]))
        if score > self.episode_best:
            self.episode_best = int(score)

    # ---- grid bookkeeping -------------------------------------------------
    def _grid_step(self, idx, old_hx, old_hy, eaten_fx, eaten_fy):
        """The few cell writes one step costs, so the grid never needs a
        rebuild. Body cells are handled by the caller, this covers the head,
        the ahead marker and the food."""
        cols, rows = self.board.cols, self.board.rows
        g = self.grids[idx]
        hx, hy = int(self.heads[idx, 0]) // BLOCK_SIZE, int(self.heads[idx, 1]) // BLOCK_SIZE
        ogx, ogy = old_hx // BLOCK_SIZE, old_hy // BLOCK_SIZE

        if 0 <= ogx < cols and 0 <= ogy < rows:
            g[1, ogy, ogx] = 0
        if 0 <= hx < cols and 0 <= hy < rows:
            g[0, hy, hx] = 1    # the new head joins the body channel
            g[1, hy, hx] = 1

        ax, ay = int(self.ahead_cells[idx, 0]), int(self.ahead_cells[idx, 1])
        if 0 <= ax < cols and 0 <= ay < rows:
            g[3, ay, ax] = 0
        nax, nay = self._straight_cell(idx)
        if 0 <= nax < cols and 0 <= nay < rows:
            g[3, nay, nax] = 1
        self.ahead_cells[idx] = (nax, nay)

        # The caller already moved the food, so mirror what foods[] says. The
        # eaten cell is no longer food, the new one is.
        efx, efy = eaten_fx // BLOCK_SIZE, eaten_fy // BLOCK_SIZE
        if 0 <= efx < cols and 0 <= efy < rows:
            g[2, efy, efx] = 0
        fx, fy = int(self.foods[idx, 0]) // BLOCK_SIZE, int(self.foods[idx, 1]) // BLOCK_SIZE
        if 0 <= fx < cols and 0 <= fy < rows:
            g[2, fy, fx] = 1


# ==========================================
# PART 2: AGENT (shared network and replay)
# ==========================================

class VecAgent:
    """One network, one replay buffer, N games feeding it."""

    def __init__(self, device, use_norm=True, obs_mode="basic", max_memory=MAX_MEMORY):
        self.device = device
        self.n_games = 0
        self.memory = deque(maxlen=max_memory)
        self.use_norm = use_norm
        self.obs_mode = obs_mode
        # Set from the environment by train(), so the network width and the
        # observation width can never disagree.
        self.obs_width = VecSnake.WIDTHS.get(obs_mode, VecSnake.WIDTHS["basic"])
        self.model = build_model(obs_mode, self.obs_width, use_norm=use_norm).to(device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=GAMMA)
        self.record = 0
        self.total_score = 0
        self.decay_episodes = float(EPSILON_DECAY_GAMES)

    def architecture(self):
        # Shared with the single game script, so a checkpoint written by either
        # one is recognised by the other.
        if self.obs_mode == "grid":
            return grid_architecture_fingerprint()
        return architecture_fingerprint(input_size=self.obs_width,
                                        use_norm=getattr(self, "use_norm", True))

    # ---- exploration ------------------------------------------------------
    def epsilon_for(self, episodes):
        """Same curve as the single game script, keyed on episodes finished.

        self.decay_episodes is scaled at startup, because 64 games finish
        episodes 64 times faster than one game. Without scaling, exploration
        would collapse to the floor within a handful of steps and the agent
        would never learn anything.
        """
        progress = min(1.0, episodes / max(1.0, self.decay_episodes))
        return max(EPSILON_MIN, EPSILON_START * ((1.0 - progress) ** 1.5))

    def act(self, obs, epsilon):
        """One batched forward pass, N actions out. All the random moves are
        decided with numpy so there is no per game Python overhead."""
        n = obs.shape[0]
        actions = np.empty(n, dtype=np.int64)
        explore = np.random.random(n) < epsilon
        actions[explore] = np.random.randint(0, 3, size=int(explore.sum()))

        exploit_idx = np.flatnonzero(~explore)
        if exploit_idx.size:
            with torch.no_grad():
                x = torch.as_tensor(obs[exploit_idx], dtype=torch.float32, device=self.device)
                q = self.model(x)
                actions[exploit_idx] = q.argmax(dim=1).cpu().numpy()
        return actions

    # ---- learning ---------------------------------------------------------
    def remember_many(self, obs, actions, rewards, next_obs, dones):
        one_hot = np.zeros((actions.shape[0], 3), dtype=np.int8)
        one_hot[np.arange(actions.shape[0]), actions] = 1
        for i in range(actions.shape[0]):
            self.memory.append((
                obs[i], one_hot[i].copy(), float(rewards[i]),
                next_obs[i], bool(dones[i]),
            ))

    def train_on_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return None
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        return None

    def remember_one(self, obs, action, reward, next_obs, done):
        """Store a single transition, used by the n step path."""
        one_hot = np.zeros(3, dtype=np.int8)
        one_hot[int(action)] = 1
        self.memory.append((obs, one_hot, float(reward), next_obs, bool(done)))

    # ---- checkpoint -------------------------------------------------------
    @staticmethod
    def _torch_load(path):
        return torch_load_checkpoint(path, DEVICE)

    def load(self, path):
        if not os.path.exists(path):
            print("=> No checkpoint found. Starting fresh.")
            return
        print("=> Loading checkpoint...")
        try:
            ck = self._torch_load(path)
            arch = ck.get("architecture")
            if arch is not None and arch != self.architecture():
                print(f"=> Checkpoint shape {arch} does not match {self.architecture()}")
                print("=> Starting fresh.")
                return
            self.model.load_state_dict(ck["model_state"])
            try:
                self.trainer.optimizer.load_state_dict(ck["optimizer_state"])
            except (ValueError, KeyError) as e:
                print(f"=> Optimizer state did not match ({e}), keeping weights only.")
            self.n_games = ck.get("n_games", 0)
            self.record = ck.get("record", 0)
            self.total_score = ck.get("total_score", 0)
            print(f"=> Resumed from {self.n_games} games with record {self.record}")
        except Exception as e:
            print(f"=> Could not load checkpoint: {type(e).__name__}: {e}. Starting fresh.")

    def save(self, path, extra=None):
        """Write the checkpoint atomically.

        A plain torch.save can leave a half written file if the process dies
        mid write, which then fails to load. Writing to a temporary file first
        and renaming means the path is always either the old file or the new
        one, never something in between.
        """
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        payload = {
            "n_games": self.n_games,
            "record": self.record,
            "total_score": self.total_score,
            "architecture": self.architecture(),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.trainer.optimizer.state_dict(),
        }
        if extra:
            payload.update(extra)
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def load_any(self, *paths):
        """Try each path in order and load the first one that exists.

        Called with the latest file first and the best file second, so a fresh
        run picks up whichever one is actually there.
        """
        for p in paths:
            if p and os.path.exists(p):
                self.load(p)
                return True
        print("=> No checkpoint found. Starting fresh.")
        return False


# ==========================================
# PART 3: TRAINING LOOP
# ==========================================

def resolve_device(name):
    if name == "auto":
        return DEVICE
    if name == "cuda" and not torch.cuda.is_available():
        print("=> cuda was asked for but torch cannot see a GPU. Using CPU.")
        print(f"   this torch build is {torch.__version__}, cuda build {torch.version.cuda}")
        print("   if it ends in +cpu, you have the CPU only wheel.")
        return torch.device("cpu")
    return torch.device(name)


def _parse_board(spec):
    """'40x30' becomes (40, 30). Keeps the CLI on one flag."""
    if spec is None:
        return None
    if isinstance(spec, (tuple, list)):
        return int(spec[0]), int(spec[1])
    cols, rows = spec.lower().split("x")
    return int(cols), int(rows)


def greedy_eval(agent, obs_mode, board, games=20, seed=4242, step_reward=0.0,
                distance_reward=0.0):
    """Score the current weights on a fresh board, greedily, no exploration.

    Every arm uses this with the SAME target board, so the numbers are
    comparable even when one arm trained on 10x10 and another on 40x30. That is
    the whole point: training scores on different boards are not comparable,
    these are. The reward shaping must match the training run, because it is
    part of the MDP the Q values were learned for.
    """
    cols, rows = board
    env = VecSnake(games, obs_mode=obs_mode, cols=cols, rows=rows,
                   step_reward=step_reward, distance_reward=distance_reward)
    np.random.seed(seed)
    obs = env.observe()
    scores = []
    was_training = agent.model.training
    agent.model.eval()
    try:
        while len(scores) < games:
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
                q = agent.model(x)
            a = q.argmax(dim=1).cpu().numpy()
            obs, _r, done, sc = env.step(a)
            for i in np.flatnonzero(done):
                scores.append(int(sc[i]))
    finally:
        if was_training:
            agent.model.train()
    return scores


def train(n_games, total_steps, train_every, device_name, checkpoint,
          best_checkpoint=VEC_BEST_CHECKPOINT, save_every=60.0, quiet=False,
          log_backend="both", run_name=None, log_dir="./runs", report_every=5.0,
          log_every=1.0, obs_mode="basic", use_norm=True,
          target_net=False, double_dqn=False, n_step=1, tau=0.01,
          patience=0, min_train_calls=200, log_group=None,
          board=None, max_memory=None,
          eval_target=None, eval_every=0, eval_games=20,
          step_reward=0.0, distance_reward=0.0):
    """Train the vectorized agent.

    Two checkpoint files are kept on purpose:

      checkpoint       the latest state, rewritten every save_every seconds
      best_checkpoint  the model that held the record score

    They are separate because the latest file is what you resume from, and the
    best file is what you show off. If they were one file, a lucky random game
    early in training could overwrite a genuinely good model.

    Curves go to TensorBoard and MLflow. Set log_backend to "none" to skip both,
    or "tensorboard" / "mlflow" for just one. quiet only silences the terminal,
    it does not stop the logging.

    Advanced methods, all off by default:

      target_net  a lagged copy computes the bootstrap target, updated by a
                  Polyak average. The standard DQN fix.
      double_dqn  the online net picks the next action, the target net scores
                  it. Needs target_net to do anything.
      n_step      sum rewards over n steps when filling the replay buffer.
                  Propagates the sparse food reward faster.

    patience    stop when the best score has not improved for this many training
                calls. 0 disables it and runs the full step budget. This is what
                keeps a sweep from spending hours on a configuration that has
                already plateaued.
    """
    device = resolve_device(device_name)
    print(f"=> {n_games} parallel games on {device}")
    if device.type == "cuda":
        print(f"=> GPU: {torch.cuda.get_device_name(0)}")

    # Readable TensorBoard folder name when the caller did not pick one, so the
    # sidebar says what the run is instead of run-20261016-011435.
    if not run_name:
        run_name = snake_run_name(
            group=log_group or "s99",
            obs_mode=obs_mode, use_norm=use_norm, target_net=target_net,
            double_dqn=double_dqn, n_step=n_step,
        )
        print(f"=> run name: {run_name}")

    board_dims = _parse_board(board)
    env = VecSnake(n_games, obs_mode=obs_mode,
                   cols=board_dims[0] if board_dims else None,
                   rows=board_dims[1] if board_dims else None,
                   step_reward=step_reward, distance_reward=distance_reward)
    agent = VecAgent(device, use_norm=use_norm, obs_mode=obs_mode,
                     max_memory=max_memory or MAX_MEMORY)
    # The network width comes from the environment. If these two ever disagree,
    # the replay samples and the model cannot be multiplied and training raises.
    agent.obs_width = env.obs_width
    agent.model = build_model(obs_mode, env.obs_width, use_norm=use_norm).to(device)
    agent.trainer = QTrainer(agent.model, lr=LR, gamma=GAMMA,
                             target=target_net, double_dqn=double_dqn,
                             n_step=n_step, tau=tau)
    if obs_mode == "grid":
        print(f"=> state: {obs_mode}, 4 channel board {env.board.cols}x{env.board.rows}, "
              f"conv net with global pooling (same weights on any board size)")
    else:
        print(f"=> state vector: {obs_mode}, {env.obs_width} features, "
              f"LayerNorm {'on' if use_norm else 'OFF'}")
    print(f"=> trainer: {agent.trainer.describe()}")
    print(f"=> training board: {env.board.cols}x{env.board.rows} "
          f"({env.board.cell_count()} cells), replay cap {agent.memory.maxlen:,}")
    eval_target_dims = _parse_board(eval_target) if eval_target else None
    if eval_target_dims:
        print(f"=> eval target: {eval_target_dims[0]}x{eval_target_dims[1]}, "
              f"{eval_games} greedy games every {eval_every:,} steps")
    if patience:
        print(f"=> early stop: no improvement for {patience} training calls "
              f"(after at least {min_train_calls})")
    # Episodes arrive n_games times faster, so the exploration curve has to be
    # stretched out or epsilon would hit the floor in a few hundred steps.
    #
    # The scaling is deliberately NOT linear in n_games. A linear stretch means
    # epsilon is still 0.94 after almost 750 finished games, which wastes the
    # whole speed advantage on random flailing. Scaling by sqrt(n_games) keeps
    # the useful part of the curve reachable while still exploring enough.
    agent.decay_episodes = EPSILON_DECAY_GAMES * (n_games ** 0.5)

    # Resume from the latest state if there is one, otherwise from the best
    # model file. They are different files now, see the comments in the loop.
    resumed = agent.load_any(checkpoint, best_checkpoint)

    # Two different step counters, and confusing them was a real bug.
    #
    #   resume_steps  how many steps happened before this run, ever
    #   steps_done    how many steps have happened IN THIS RUN
    #
    # --steps is a budget for THIS RUN, not a cumulative target. It used to be
    # treated as cumulative, so resuming from a 2,000,128 step checkpoint with
    # --steps 1500000 ran ZERO steps and exited instantly with an absurd
    # throughput number. The curves use the cumulative axis so they line up
    # across runs; the loop and the speed numbers use the per run counter.
    resume_steps = 0
    episode_offset = 0
    if resumed:
        try:
            prev = agent._torch_load(checkpoint if os.path.exists(checkpoint) else best_checkpoint)
            resume_steps = int(prev.get("steps", 0))
            episode_offset = int(agent.n_games)
            print(f"=> resuming after {resume_steps:,} previous steps, "
                  f"running {total_steps:,} more")
        except Exception:
            resume_steps = 0
            episode_offset = 0

    steps_done = 0
    total_seen = resume_steps

    # ---- logging ---------------------------------------------------------
    if log_backend == "none":
        log = NullLogger()
    else:
        try:
            log = Logger(
                backend=log_backend,
                run_name=run_name,
                log_dir=log_dir,
                params={
                    "games": n_games,
                    "total_steps": total_steps,
                    "train_every": train_every,
                    "obs_mode": obs_mode,
                    "obs_width": env.obs_width,
                    "train_board": f"{env.board.cols}x{env.board.rows}",
                    "use_norm": use_norm,
                    "hidden_size": HIDDEN_SIZE,
                    "hidden_layers": HIDDEN_LAYERS,
                    "lr": LR,
                    "gamma": GAMMA,
                    "batch_size": BATCH_SIZE,
                    "max_memory": MAX_MEMORY,
                    "epsilon_min": EPSILON_MIN,
                    "epsilon_decay_episodes": agent.decay_episodes,
                    "reward_food": REWARD_FOOD,
                    "reward_death": REWARD_DEATH,
                    "device": str(device),
                    "torch_threads": torch.get_num_threads(),
                    "resumed_from_step": resume_steps,
                    "max_memory": agent.memory.maxlen,
                    "step_reward": step_reward,
                    "distance_reward": distance_reward,
                    "eval_target": (f"{eval_target_dims[0]}x{eval_target_dims[1]}"
                                    if eval_target_dims else None),
                },
            )
        except Exception as e:
            print(f"=> WARNING: logging disabled ({type(e).__name__}: {e})")
            log = NullLogger()

    if not quiet:
        print(f"=> epsilon decays over {agent.decay_episodes:,.0f} episodes "
              f"({EPSILON_DECAY_GAMES} x sqrt({n_games}))")

    obs = env.observe()
    next_train = train_every
    t0 = time.perf_counter()
    last_report = t0
    last_log = t0
    last_save = t0
    last_eval = 0
    report_step = steps_done
    log_step = steps_done
    train_calls = 0
    diag_acc = {}
    interrupted = False
    stopped_early = False

    # n-step bookkeeping. The reward is summed over n steps, then the whole
    # batch is stored at once against the observations that opened the window.
    #
    # This is deliberately NOT per game gating. Gating per game stored only
    # 1/n as many samples, so the replay buffer filled 1/n as fast, so training
    # started far later and the agent learned nothing. Every n-step run in the
    # first sweep scored 1 or 2 because of exactly that. Comparing n-step needs
    # the buffer to fill at the same rate, so the flush is on a shared counter
    # and it fires for all games together.
    pending_r = np.zeros(n_games, dtype=np.float32)
    track_obs = obs.copy()
    track_act = np.zeros(n_games, dtype=np.int64)
    steps_in_window = 0

    # Early stopping state. A new record resets the counter, so a configuration
    # that is still improving keeps running and one that has stalled stops.
    calls_since_best = 0
    best_seen = agent.record

    try:
        while steps_done < total_steps:
            eps = agent.epsilon_for(env.episodes)
            actions = agent.act(obs, eps)

            next_obs, rewards, done, scores = env.step(actions)

            if n_step > 1:
                # The window is (track_obs, track_act) at the start, and the
                # reward is the sum over the whole window.
                #
                # track_obs and track_act must stay untouched for the whole
                # window. An earlier version overwrote track_act with the LATEST
                # action at flush time, so a sample paired an observation with an
                # action that did not produce its reward. Every n-step run scored
                # 1 or 2 because of that, and it silently corrupted the data
                # rather than crashing.
                pending_r += rewards
                steps_in_window += 1

                # A game that ended closes its own window, whatever its length.
                # Without this a game that dies before the window fills
                # contributes nothing, which starved the buffer: replay size
                # 41,413 instead of 100,000 in the first sweep.
                ended = np.flatnonzero(done)
                if ended.size and steps_in_window < n_step:
                    for i in ended:
                        agent.remember_one(track_obs[i], track_act[i],
                                           float(pending_r[i]), next_obs[i], True)
                        pending_r[i] = 0.0

                if steps_in_window >= n_step:
                    agent.remember_many(track_obs, track_act, pending_r,
                                        next_obs, done)
                    pending_r = np.zeros(n_games, dtype=np.float32)
                    # The next window starts at the state we just reached.
                    track_obs = next_obs.copy()
                    track_act = actions.copy()
                    steps_in_window = 0
            else:
                # next_obs is the post reset observation for finished games. That
                # is fine here because done=True makes the bootstrap term fall
                # away in the trainer, so the next state is never read.
                agent.remember_many(obs, actions, rewards, next_obs, done)

            # Score keeping for the games that just ended.
            for i in np.flatnonzero(done):
                s = int(scores[i])
                # Set the record before n_games moves, so a checkpoint saved
                # right here carries a game count that matches its score.
                if s > agent.record:
                    # A new record goes to its own file. The latest file is
                    # written on a timer below, so a lucky random game that sets
                    # a high score early cannot cost you the good model.
                    agent.record = s
                    agent.save(best_checkpoint, extra={"steps": total_seen})
                    log.add("episode/best_score", s, episode_offset + agent.n_games + 1)
                    print(f"  [record] {s} at episode {episode_offset + agent.n_games + 1}, "
                          f"step {steps_done:,} into this run")
                agent.n_games += 1
                agent.total_score += s
                # Log every finished episode against its own episode counter, so
                # the score curve is per episode rather than an average that
                # hides the spread.
                log.add("episode/score", s, episode_offset + agent.n_games)

            obs = next_obs
            steps_done += n_games
            total_seen += n_games

            # Periodic greedy evaluation on the TARGET board. This is the fair
            # comparison across arms: a run trained on 10x10 and one trained on
            # 40x30 both get scored here on the same 40x30 board, so the curves
            # can be overlaid without lying about what they mean.
            if eval_target_dims and eval_every and steps_done - last_eval >= eval_every:
                last_eval = steps_done
                t_ev = time.perf_counter()
                ev_scores = greedy_eval(agent, obs_mode, eval_target_dims,
                                        games=eval_games,
                                        step_reward=step_reward,
                                        distance_reward=distance_reward)
                log.add_many({
                    "eval/score_mean": float(np.mean(ev_scores)),
                    "eval/score_best": float(max(ev_scores)),
                    "eval/score_min": float(min(ev_scores)),
                    "eval/wall_clock_s": time.perf_counter() - t_ev,
                }, total_seen)
                print(f"  [eval {eval_target_dims[0]}x{eval_target_dims[1]}] "
                      f"mean {np.mean(ev_scores):.2f} best {max(ev_scores)} "
                      f"({eval_games} games, step {total_seen:,})")

            if steps_done >= next_train:
                agent.train_on_replay()
                train_calls += 1
                next_train += train_every

                # Early stopping. A new record resets the counter, so a run that
                # is still improving continues and one that has flattened out
                # stops instead of burning hours for nothing.
                if patience and train_calls >= min_train_calls:
                    if agent.record > best_seen:
                        best_seen = agent.record
                        calls_since_best = 0
                    else:
                        calls_since_best += 1
                        if calls_since_best >= patience:
                            print(f"=> EARLY STOP: best score stuck at {agent.record} "
                                  f"for {calls_since_best} training calls "
                                  f"({steps_done:,} steps this run)")
                            log.add("early_stop/triggered", 1, total_seen)
                            stopped_early = True
                            # Make sure a best file exists before leaving. If no
                            # record was ever set there is nothing else to save,
                            # and a run with no checkpoint cannot be resumed.
                            if not os.path.exists(best_checkpoint):
                                agent.save(best_checkpoint,
                                           extra={"steps": total_seen,
                                                  "stopped_early": True})
                                print(f"=> saved {best_checkpoint} at exit")
                            break
                log.add("early_stop/calls_since_best", calls_since_best, total_seen)
                # Diagnostics are ACCUMULATED here and averaged into one point
                # per report below. Writing them every step produced 49,000 rows
                # per metric in a 150,000 step run, which makes the MLflow store
                # enormous and slow to open. The averaged curve shows the same
                # trend.
                tr = agent.trainer
                for key, value in (
                    ("loss", tr.last_loss),
                    ("grad_norm", tr.last_grad_norm),
                    ("q_mean", tr.last_q_mean),
                    ("q_max", tr.last_q_max),
                    ("target_mean", tr.last_target_mean),
                ):
                    if value is not None:
                        diag_acc.setdefault(key, []).append(float(value))

            now = time.perf_counter()

            # Time based save, so a plateau in the score cannot cost an hour of
            # work. Records stop falling late in training, and without this the
            # only save points would be the rare new record.
            if now - last_save >= save_every:
                agent.save(checkpoint, extra={"steps": total_seen,
                                            "stopped_early": stopped_early})
                last_save = now
                print(f"  [saved] {steps_done:,} into this run "
                      f"({total_seen:,} total), best {agent.record}")

            # Scalar block for the CURVES, on its own shorter cadence.
            #
            # This is deliberately separate from the console report below. When
            # both were on the same 10 second timer, a short run logged only two
            # points per metric, which is not a curve. The curves want frequent
            # points; the terminal only wants to be readable.
            if now - last_log >= log_every:
                sps = (steps_done - log_step) / (now - last_log)
                log_step = steps_done
                recent_r = env.episode_returns
                recent_s = env.episode_scores
                recent_l = env.episode_lengths

                log.add_many({
                    "progress/steps": total_seen,
                    "progress/run_steps": steps_done,
                    "progress/episodes": episode_offset + agent.n_games,
                    "progress/wall_clock_s": now - t0,
                    "throughput/steps_per_s": sps,
                    "explore/epsilon": eps,
                    "replay/size": len(agent.memory),
                    "replay/train_calls": train_calls,
                    "episode/return_mean_200": float(np.mean(recent_r)) if recent_r else 0.0,
                    "episode/score_mean_200": float(np.mean(recent_s)) if recent_s else 0.0,
                    "episode/score_max_200": float(max(recent_s)) if recent_s else 0.0,
                    "episode/length_mean_200": float(np.mean(recent_l)) if recent_l else 0.0,
                    "episode/best_score": agent.record,
                }, steps_done)

                # One averaged point per diagnostic per log window, instead of
                # one per training step.
                if diag_acc:
                    log.add_many(
                        {f"train/{k}_mean": float(np.mean(v)) for k, v in diag_acc.items() if v},
                        steps_done,
                    )
                    diag_acc.clear()
                log.maybe_flush()
                last_log = now

            # Console report, on its own slower cadence.
            if not quiet and now - last_report >= report_every:
                sps_print = (steps_done - report_step) / (now - last_report)
                recent_r = env.episode_returns
                avg_ret = float(np.mean(recent_r)) if recent_r else 0.0
                print(
                    f"steps {steps_done:>10,} | games {agent.n_games:>7,} | "
                    f"eps {eps:.3f} | best {agent.record:>3} | "
                    f"avg return {avg_ret:6.2f} | {sps_print:8,.0f} steps/s | "
                    f"replay {len(agent.memory):>6,}"
                )
                last_report = now
                report_step = steps_done
    except KeyboardInterrupt:
        interrupted = True
        print("\n=> Stopped by user. Saving before exit...")
        agent.save(checkpoint, extra={"steps": total_seen,
                                            "stopped_early": stopped_early})
        print(f"=> Saved {checkpoint}")

    elapsed = time.perf_counter() - t0
    # Always leave a current latest file behind, whether we finished the run or
    # the user stopped it.
    agent.save(checkpoint, extra={"steps": total_seen,
                                            "stopped_early": stopped_early})

    recent_r = env.episode_returns
    recent_s = env.episode_scores
    recent_l = env.episode_lengths
    sps = steps_done / max(elapsed, 1e-9)

    log.add_many({
        "progress/steps": total_seen,
        "progress/run_steps": steps_done,
        "progress/episodes": episode_offset + agent.n_games,
        "progress/wall_clock_s": elapsed,
        "throughput/steps_per_s": sps,
        "episode/best_score": agent.record,
        "episode/return_mean_200": float(np.mean(recent_r)) if recent_r else 0.0,
        "episode/score_mean_200": float(np.mean(recent_s)) if recent_s else 0.0,
        "episode/length_mean_200": float(np.mean(recent_l)) if recent_l else 0.0,
    }, total_seen)
    # The final eval on the target board, so every run ends with the number the
    # comparison is actually about, whatever board it trained on.
    if eval_target_dims:
        print(f"=> final eval on {eval_target_dims[0]}x{eval_target_dims[1]}...")
        ev_scores = greedy_eval(agent, obs_mode, eval_target_dims, games=eval_games,
                                step_reward=step_reward,
                                distance_reward=distance_reward)
        log.add_many({
            "eval/final_mean": float(np.mean(ev_scores)),
            "eval/final_best": float(max(ev_scores)),
            "eval/final_min": float(min(ev_scores)),
        }, total_seen)
        print(f"=> final eval: mean {np.mean(ev_scores):.2f}, "
              f"best {max(ev_scores)}, min {min(ev_scores)} "
              f"({len(ev_scores)} games)")

    add_histogram(log, "episode/return_hist", list(recent_r), total_seen)
    add_histogram(log, "episode/score_hist", list(recent_s), total_seen)
    log.close()

    print("-" * 72)
    print(f"{'stopped' if interrupted else 'finished'} after {steps_done:,} steps "
          f"in this run, {total_seen:,} total, in {elapsed:.1f}s")
    print(f"  throughput     : {sps:,.0f} steps/s")
    print(f"  games finished : {agent.n_games}")
    print(f"  best score     : {agent.record}")
    print(f"  avg return     : {float(np.mean(recent_r)) if recent_r else 0.0:.2f} (last {len(recent_r)} games)")
    print(f"  replay size    : {len(agent.memory):,}")
    print(f"  train calls    : {train_calls:,}")
    print(f"  latest file    : {checkpoint}")
    print(f"  best file      : {best_checkpoint}")
    return sps


def main():
    p = argparse.ArgumentParser(description="Vectorized Snake DQN")
    p.add_argument("--games", type=int, default=64, help="parallel games (default 64)")
    p.add_argument("--steps", type=int, default=500_000,
                   help="steps to run THIS TIME, not a lifetime total. Resuming "
                        "from a 2 million step checkpoint with --steps 1500000 "
                        "runs 1.5 million MORE steps")
    p.add_argument("--train-every", type=int, default=8,
                   help="run one replay batch every N steps (default 8)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "xpu"])
    p.add_argument("--checkpoint", default=VEC_CHECKPOINT,
                   help="latest state, rewritten on a timer and on Ctrl+C")
    p.add_argument("--best-checkpoint", default=VEC_BEST_CHECKPOINT,
                   help="model that held the record score, kept separately")
    p.add_argument("--save-every", type=float, default=60.0,
                   help="seconds between latest state saves (default 60)")
    p.add_argument("--log", dest="log_backend", default="both",
                   choices=["both", "tensorboard", "mlflow", "none"],
                   help="where the curves go (default both)")
    p.add_argument("--run-name", default=None,
                   help="name for this run, defaults to run-<timestamp>")
    p.add_argument("--log-dir", default="./runs",
                   help="folder for tensorboard and mlflow output (default ./runs)")
    p.add_argument("--report-every", type=float, default=5.0,
                   help="seconds between console reports (default 5)")
    p.add_argument("--log-every", type=float, default=1.0,
                   help="seconds between logged points on the curves (default 1)")
    p.add_argument("--state", dest="obs_mode", default="basic",
                   choices=["basic", "rich", "flood", "grid"],
                   help="what the agent can see: basic 11 features, rich 21, "
                        "flood 25 trap features, or grid, the raw 4 channel "
                        "board read by a conv net that works on any board size")
    p.add_argument("--board", default=None,
                   help="training board in cells, e.g. 40x30 or 12x12 "
                        "(default 40x30)")
    p.add_argument("--max-memory", type=int, default=None,
                   help="replay cap. Default 100000, but a grid state on 40x30 "
                        "is 4800 bytes per sample, so large boards need less")
    p.add_argument("--eval-target", default=None,
                   help="board to score greedy games on, e.g. 40x30. This is "
                        "the fair cross arm comparison")
    p.add_argument("--eval-every", type=int, default=0,
                   help="steps between target board evals. 0 means only at the end")
    p.add_argument("--eval-games", type=int, default=20,
                   help="greedy games per eval")
    p.add_argument("--step-reward", type=float, default=0.0,
                   help="reward on steps that do not eat. Negative values "
                        "penalise circling, which is the fix for the conv net "
                        "learning to loop until the starvation guard kills it")
    p.add_argument("--distance-reward", type=float, default=0.0,
                   help="shaping: reward for moving closer to the food, penalty "
                        "for moving away, per cell. Gives the conv net the food "
                        "gradient that the 11 features hand to the DQN for free")
    p.add_argument("--no-norm", dest="use_norm", action="store_false",
                   help="drop LayerNorm from the hidden layers. Tested: does not help")
    p.set_defaults(use_norm=True)
    p.add_argument("--target-net", action="store_true",
                   help="lagged target network for the bootstrap, Polyak averaged")
    p.add_argument("--double-dqn", action="store_true",
                   help="online net picks the action, target net scores it")
    p.add_argument("--n-step", type=int, default=1,
                   help="EXPERIMENTAL AND NOT WORKING. Every configuration tried "
                        "scores 1 or 2 against 50 for the baseline, and the cause "
                        "is not found. Left in place to be debugged, do not use "
                        "it for a comparison")
    p.add_argument("--tau", type=float, default=0.01,
                   help="Polyak rate for the target network (default 0.01)")
    p.add_argument("--patience", type=int, default=0,
                   help="stop after this many training calls with no new record. "
                        "0 disables early stopping")
    p.add_argument("--min-train-calls", type=int, default=200,
                   help="do not early stop before this many training calls")
    p.add_argument("--log-group", default=None,
                   help="prefix for the auto run name, e.g. s03")
    p.add_argument("--quiet", action="store_true",
                   help="silence the terminal, logging still happens")
    a = p.parse_args()
    train(a.games, a.steps, a.train_every, a.device, a.checkpoint,
          a.best_checkpoint, a.save_every, a.quiet, a.log_backend,
          a.run_name, a.log_dir, a.report_every, a.log_every, a.obs_mode,
          a.use_norm, a.target_net, a.double_dqn, a.n_step, a.tau,
          a.patience, a.min_train_calls, a.log_group,
          board=a.board, max_memory=a.max_memory,
          eval_target=a.eval_target, eval_every=a.eval_every,
          eval_games=a.eval_games, step_reward=a.step_reward,
          distance_reward=a.distance_reward)


if __name__ == "__main__":
    main()
