"""What is actually achievable on this Snake board?

The question that matters
------------------------
Public Snake projects that score very high are almost never learned from
scratch. They use SEARCH: BFS to the food with a safety check, a Hamiltonian
cycle, or A* with a survival heuristic. Those are planning algorithms with the
whole board visible, and they beat any of our learned agents easily.

This file measures that directly on OUR environment, so the comparison is
apples to apples and not a claim about someone else's code.

Policies, all hand written, no learning:
  random        pick any legal move
  greedy        move towards the food, ignore danger
  flood         among legal moves, take the one leaving the most reachable space
  bfs_safe      find the shortest path to the food, take the first step, but
                only if the tail is still reachable after eating. Otherwise
                follow the tail.

Usage
-----
    python tools/search_baseline.py --games 30
    python tools/search_baseline.py --games 30 --board 20x15
"""

import argparse
import os
import sys
import time
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

from snake_core import BLOCK_SIZE, Direction


class Board:
    """Cells, not pixels. Search is much easier to write in cell space."""

    def __init__(self, cols=40, rows=30):
        self.cols = cols
        self.rows = rows
        self.reset()

    def reset(self):
        cx, cy = self.cols // 2, self.rows // 2
        self.snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.direction = Direction.RIGHT
        self.score = 0
        self.steps = 0
        self.place_food()

    def place_food(self):
        free = [(x, y) for x in range(self.cols) for y in range(self.rows)
                if (x, y) not in self.snake]
        self.food = free[np.random.randint(len(free))] if free else None

    def legal_moves(self):
        head = self.snake[0]
        clock = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        i = clock.index(self.direction)
        out = {}
        for name, idx in (("straight", i), ("right", (i + 1) % 4), ("left", (i - 1) % 4)):
            d = clock[idx]
            dx, dy = {Direction.RIGHT: (1, 0), Direction.LEFT: (-1, 0),
                      Direction.UP: (0, -1), Direction.DOWN: (0, 1)}[d]
            cell = (head[0] + dx, head[1] + dy)
            wall = not (0 <= cell[0] < self.cols and 0 <= cell[1] < self.rows)
            # The tail moves away, so entering the current tail cell is fine.
            body = set(list(self.snake)[1:-1])
            out[name] = (d, cell, wall or cell in body)
        return out

    def step(self, name):
        moves = self.legal_moves()
        d, cell, dead = moves[name]
        self.direction = d
        self.steps += 1
        if dead:
            return -10.0, True
        if self.steps > 100 * len(self.snake):
            return -10.0, True
        self.snake.appendleft(cell)
        if cell == self.food:
            self.score += 1
            self.place_food()
            return 10.0, False
        self.snake.pop()
        return 0.0, False

    # ---- helpers for the planners ---------------------------------------
    def occupancy(self, drop_tail=True):
        body = set(self.snake)
        if drop_tail and len(self.snake) > 1:
            body.discard(self.snake[-1])
        return body

    def reachable(self, start, blocked):
        if start in blocked:
            return set()
        seen = {start}
        q = [start]
        while q:
            x, y = q.pop()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if not (0 <= nx < self.cols and 0 <= ny < self.rows):
                    continue
                if (nx, ny) in seen or (nx, ny) in blocked:
                    continue
                seen.add((nx, ny))
                q.append((nx, ny))
        return seen

    def bfs_first_step(self, blocked, target):
        """First move of a shortest path from the head to target."""
        start = self.snake[0]
        if target is None or start == target:
            return None
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == target:
                break
            for nxt in ((cur[0] + 1, cur[1]), (cur[0] - 1, cur[1]),
                        (cur[0], cur[1] + 1), (cur[0], cur[1] - 1)):
                if not (0 <= nxt[0] < self.cols and 0 <= nxt[1] < self.rows):
                    continue
                if nxt in prev or nxt in blocked:
                    continue
                prev[nxt] = cur
                q.append(nxt)
        if target not in prev:
            return None
        node = target
        while prev[node] is not None and prev[prev[node]] is not None:
            node = prev[node]
        return node


# ==========================================
# POLICIES
# ==========================================

def policy_random(b):
    return np.random.choice(["straight", "right", "left"])


def policy_greedy(b):
    head = b.snake[0]
    food = b.food
    moves = b.legal_moves()
    best, best_d = None, 10 ** 9
    for name, (d, cell, dead) in moves.items():
        if dead:
            continue
        dist = abs(cell[0] - food[0]) + abs(cell[1] - food[1])
        if dist < best_d:
            best, best_d = name, dist
    if best is None:
        return "straight"
    return best


def policy_flood(b):
    """Among legal moves, take the one leaving the most reachable space.

    The first version of this scored 0. It kept a best_area counter and compared
    a tuple against it, which never selected anything, so the snake turned in
    place until the starvation guard killed it. Selection is on the tuple now.
    """
    moves = b.legal_moves()
    blocked = b.occupancy()
    food = b.food
    best_name = None
    best_key = None
    for name, (d, cell, dead) in moves.items():
        if dead:
            continue
        area = len(b.reachable(cell, blocked))
        dist = abs(cell[0] - food[0]) + abs(cell[1] - food[1])
        # Space first, then closeness to the food as the tie break, so it still
        # eats instead of circling in open space until it starves.
        key = (area, -dist)
        if best_key is None or key > best_key:
            best_key = key
            best_name = name
    return best_name if best_name is not None else "straight"


def policy_bfs_safe(b):
    """Shortest path to the food, but only when it stays safe.

    The safety check is what makes this work: after eating, the tail must still
    be reachable. If it is not, the snake would seal itself in, so instead it
    follows its tail to buy time.
    """
    moves = b.legal_moves()
    blocked = b.occupancy()

    if b.food is not None:
        first = b.bfs_first_step(blocked, b.food)
        if first is not None:
            for name, (d, cell, dead) in moves.items():
                if dead or cell != first:
                    continue
                # Would eating here trap it? Simulate the body after the move.
                body_after = set(b.snake)
                body_after.discard(b.snake[-1])
                body_after.add(cell)
                grows = cell == b.food
                new_head = cell
                # After eating, the tail does not move, so it is a wall.
                block_after = set(body_after)
                if grows:
                    block_after.discard(b.snake[-1])
                else:
                    block_after.discard(b.snake[-1])
                block_after.discard(new_head)
                area_after = b.reachable(new_head, block_after)
                tail = b.snake[-1]
                if tail in area_after or len(area_after) > len(b.snake):
                    return name

    # Fall back to following the tail, which is the standard survival move.
    tail = b.snake[-1]
    first = b.bfs_first_step(blocked, tail)
    if first is not None:
        for name, (d, cell, dead) in moves.items():
            if not dead and cell == first:
                return name

    # Nothing safe, take any legal move, else straight.
    for name, (d, cell, dead) in moves.items():
        if not dead:
            return name
    return "straight"


POLICIES = {
    "random": policy_random,
    "greedy": policy_greedy,
    "flood": policy_flood,
    "bfs_safe": policy_bfs_safe,
}


def play(policy, games, cols, rows, max_steps):
    scores, lengths, times = [], [], []
    for _ in range(games):
        b = Board(cols, rows)
        t0 = time.perf_counter()
        while b.steps < max_steps:
            move = policy(b)
            _, done = b.step(move)
            if done:
                break
        times.append(time.perf_counter() - t0)
        scores.append(b.score)
        lengths.append(b.steps)
    return scores, lengths, times


def main():
    p = argparse.ArgumentParser(description="Search baselines for Snake")
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--board", default="40x30", help="cols x rows")
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--policies", default="random,greedy,flood,bfs_safe")
    a = p.parse_args()

    cols, rows = (int(v) for v in a.board.lower().split("x"))
    names = [n.strip() for n in a.policies.split(",") if n.strip()]

    print("=" * 84)
    print(f"HAND WRITTEN POLICIES, no learning | board {cols}x{rows} "
          f"({cols*rows} cells) | {a.games} games each")
    print("=" * 84)
    print("These are planning algorithms. They see the whole board, which our")
    print("learned agents cannot. This sets the bar the learned agents are")
    print("being asked to reach.")
    print()

    results = []
    for name in names:
        fn = POLICIES.get(name)
        if fn is None:
            print(f"  unknown policy {name}")
            continue
        np.random.seed(1234)
        scores, lengths, times = play(fn, a.games, cols, rows, a.max_steps)
        results.append((name, np.mean(scores), max(scores), np.mean(lengths),
                        np.mean(times) * 1000))
        print(f"  {name:10s} mean {np.mean(scores):>7.2f}  best {max(scores):>5}  "
              f"mean moves {np.mean(lengths):>8.0f}  {np.mean(times)*1000:>7.1f} ms per game")

    print()
    print("=" * 84)
    print("COMPARISON")
    print("=" * 84)
    scores_only = [r[1] for r in results]
    print(f"  {'policy':12s} {'mean score':>11} {'best':>6} {'ms per game':>12}")
    for name, mean, best, _len, ms in sorted(results, key=lambda r: -r[1]):
        print(f"  {name:12s} {mean:>11.2f} {best:>6} {ms:>12.1f}")
    print()
    print("  For reference, our learned agents, greedy play, same board:")
    print("    DQN baseline 1.2M steps, 3 seeds   best 56 to 60, greedy mean 32.4")
    print("    your record 77 checkpoint          greedy mean 30.9, best 67")
    print("    LightGBM at 600k steps             best 49")
    if results:
        top = max(results, key=lambda r: r[1])
        print()
        print(f"  best hand written policy: {top[0]} at mean {top[1]:.2f}")


if __name__ == "__main__":
    main()
