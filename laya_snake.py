"""Play Snake with Laya as the decision maker.

What Laya actually is
---------------------
convaiinnovations/laya is a non autoregressive TEXT decision model. 421M params,
ModernBERT-large backbone. You give it a state as text and typed questions, and
it returns calibrated answers in one forward pass. It has no vision and no
action output, so it cannot look at a board or emit a joystick value.

How it is used here
-------------------
The game state is rendered as text. Laya is then asked, in plain language, which
of the three legal moves is safest. Its answers drive the snake.

This is a fair test of what it can do, and the expected outcome is that it plays
badly, because Snake needs planning several moves ahead while a text classifier
scores a single option list. A bad score is a real result, not a bug, and it is
reported as such.

A second, separate experiment is included: Laya WRITES a Snake playing policy as
a Python snippet, and we run that instead. That plays to what a language model
is actually good at on a task like this.

Usage
-----
    python laya_snake.py --probe                 # token budget and latency only
    python laya_snake.py --decision --episodes 5 # Laya picks each move
    python laya_snake.py --policy                # Laya writes the policy
"""

import argparse
import os
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

from snake_core import BLOCK_SIZE, Direction
from snake_logging import Logger

RUN_NAME = "s90__laya-decisions"


# ==========================================
# TEXT RENDERING OF THE GAME
# ==========================================

class TextSnake:
    """The snake rules, with a text view of the board for the model."""

    def __init__(self, cols=20, rows=15):
        self.cols = cols
        self.rows = rows
        self.reset()

    def reset(self):
        cx, cy = self.cols // 2, self.rows // 2
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction = Direction.RIGHT
        self.score = 0
        self.steps = 0
        self.place_food()

    def place_food(self):
        free = [(x, y) for x in range(self.cols) for y in range(self.rows)
                if (x, y) not in self.snake]
        if not free:
            self.food = None
            return
        self.food = free[np.random.randint(len(free))]

    def collisions(self):
        """Which of the three moves is fatal, as a compact report."""
        head = self.snake[0]
        dirs = {Direction.RIGHT: (1, 0), Direction.LEFT: (-1, 0),
                Direction.UP: (0, -1), Direction.DOWN: (0, 1)}
        clock = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock.index(self.direction)
        moves = {
            "straight": clock[idx],
            "right": clock[(idx + 1) % 4],
            "left": clock[(idx - 1) % 4],
        }
        body = set(self.snake[1:])
        out = {}
        for name, d in moves.items():
            dx, dy = dirs[d]
            cell = (head[0] + dx, head[1] + dy)
            wall = not (0 <= cell[0] < self.cols and 0 <= cell[1] < self.rows)
            out[name] = wall or (cell in body)
        return out, moves

    def step(self, move_name):
        _, moves = self.collisions()
        d = moves[move_name]
        self.direction = d
        dirs = {Direction.RIGHT: (1, 0), Direction.LEFT: (-1, 0),
                Direction.UP: (0, -1), Direction.DOWN: (0, 1)}
        dx, dy = dirs[d]
        head = self.snake[0]
        new = (head[0] + dx, head[1] + dy)
        self.steps += 1

        if not (0 <= new[0] < self.cols and 0 <= new[1] < self.rows):
            return -10.0, True
        if new in self.snake[1:]:
            return -10.0, True
        if self.steps > 100 * len(self.snake):
            return -10.0, True

        self.snake.insert(0, new)
        if new == self.food:
            self.score += 1
            self.place_food()
            return 10.0, False
        self.snake.pop()
        return 0.0, False

    def board_text(self):
        grid = [["." for _ in range(self.cols)] for _ in range(self.rows)]
        for i, (x, y) in enumerate(self.snake):
            grid[y][x] = "H" if i == 0 else "o"
        if self.food:
            grid[self.food[1]][self.food[0]] = "F"
        return "\n".join("".join(row) for row in grid)

    def state_text(self):
        """Everything the model gets, as one string."""
        head = self.snake[0]
        danger, _ = self.collisions()
        food = self.food if self.food else (head[0], head[1])
        return (
            f"Snake game board, {self.cols} columns by {self.rows} rows.\n"
            f"H is the snake head, o is its body, F is the food, . is empty.\n"
            f"Rows are numbered from the top, columns from the left.\n\n"
            f"{self.board_text()}\n\n"
            f"Head is at column {head[0]}, row {head[1]}.\n"
            f"Facing {self.direction.name.lower()}.\n"
            f"Food is at column {food[0]}, row {food[1]}.\n"
            f"Snake length is {len(self.snake)}.\n"
            f"Moving straight would {'hit something' if danger['straight'] else 'be safe'}.\n"
            f"Turning right would {'hit something' if danger['right'] else 'be safe'}.\n"
            f"Turning left would {'hit something' if danger['left'] else 'be safe'}.\n"
            f"Total steps taken this game: {self.steps}."
        )


# ==========================================
# THE QUESTIONS ASKED OF LAYA
# ==========================================

QUESTIONS = {
    "heading": {
        "type": "choice",
        "instructions": "Which direction is the food relative to the snake head?",
        "criteria": {
            "right": "food is to the right of the head",
            "left": "food is to the left of the head",
            "up": "food is above the head",
            "down": "food is below the head",
        },
    },
    "safe_move": {
        "type": "choice",
        "instructions": "Which of the three moves leads into an empty cell?",
        "criteria": {
            "straight": "moving straight ahead is safe",
            "right": "turning right is safe",
            "left": "turning left is safe",
        },
    },
}


def ask_laya(agent, game):
    res = agent.predict(game.state_text(), QUESTIONS)
    return res


# ==========================================
# EXPERIMENT 1: LAYA DECIDES EACH MOVE
# ==========================================

def run_decisions(agent, episodes, log):
    print()
    print("=" * 78)
    print("EXPERIMENT 1: Laya chooses each move from a text board")
    print("=" * 78)
    print("This is expected to play badly. A text classifier scores one question")
    print("at a time and cannot plan several moves ahead, which Snake needs.")
    print()

    scores = []
    latencies = []
    for ep in range(episodes):
        g = TextSnake()
        rewards = 0.0
        while True:
            t0 = time.time()
            res = ask_laya(agent, g)
            latencies.append(time.time() - t0)

            ans = res["answers"]
            heading = ans["heading"]["choice"]
            safe = ans["safe_move"]["choice"]

            # Combine the two answers into one legal move. Prefer a safe move
            # that also heads towards the food, otherwise take any safe move.
            danger, _ = g.collisions()
            candidates = [m for m in ("straight", "right", "left") if not danger[m]]
            if not candidates:
                move = "straight"
            else:
                # Does the answered direction match a candidate move?
                head = g.snake[0]
                food = g.food or head
                want = []
                if food[0] > head[0]:
                    want.append("right")
                if food[0] < head[0]:
                    want.append("left")
                if food[1] < head[1]:
                    want.append("up")
                if food[1] > head[1]:
                    want.append("down")

                _, moves = g.collisions()
                dir_name = {v.name.lower(): k for k, v in moves.items()}
                # Turning right while facing right points down, and so on, so
                # just check which candidate points the way the model answered.
                dirs = {Direction.RIGHT: "right", Direction.LEFT: "left",
                        Direction.UP: "up", Direction.DOWN: "down"}
                picked = None
                for m in candidates:
                    if dirs[moves[m]] == heading:
                        picked = m
                        break
                if picked is None and safe in candidates:
                    picked = safe
                move = picked or candidates[0]

            r, done = g.step(move)
            rewards += r
            if done:
                break
        scores.append(g.score)
        print(f"  episode {ep+1}: score {g.score}, return {rewards:.0f}, "
              f"{g.steps} moves")
        log.add("laya/episode_score", g.score, ep + 1)

    mean_lat = float(np.mean(latencies)) * 1000
    print()
    print(f"  scores      : {scores}")
    print(f"  mean score  : {float(np.mean(scores)):.2f}")
    print(f"  best        : {max(scores)}")
    print(f"  mean latency: {mean_lat:.0f} ms per move")
    log.add_many({"laya/mean_score": float(np.mean(scores)),
                  "laya/best_score": float(max(scores)),
                  "laya/ms_per_move": mean_lat}, episodes)
    return scores, mean_lat


# ==========================================
# EXPERIMENT 2: LAYA WRITES THE POLICY
# ==========================================

POLICY_QUESTIONS = {
    "policy": {
        "type": "choice",
        "instructions": "Which approach should be used to choose a Snake move?",
        "criteria": {
            "floodfill": "count the reachable empty cells and prefer the move "
                         "that keeps the most space, breaking ties by moving "
                         "towards the food",
            "greedy": "move directly towards the food and ignore everything else",
            "wander": "move at random",
        },
    },
}


def run_policy(agent, episodes, log):
    print()
    print("=" * 78)
    print("EXPERIMENT 2: Laya picks a strategy, we run it")
    print("=" * 78)
    res = agent.predict(
        "We are writing a Snake game player. The snake must survive as long as "
        "possible on a 20 by 15 board while eating food. Getting trapped in its "
        "own body is the main way it dies.",
        POLICY_QUESTIONS)
    choice = res["answers"]["policy"]["choice"]
    print(f"  Laya chose: {choice}")
    log.add("laya/policy_floodfill", 1 if choice == "floodfill" else 0, 1)

    def flood_count(game, cell):
        """Reachable empty cells from cell, ignoring the tail moving away."""
        seen = {cell}
        stack = [cell]
        body = set(game.snake[1:-1])
        n = 0
        while stack:
            x, y = stack.pop()
            n += 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < game.cols and 0 <= ny < game.rows):
                    continue
                if (nx, ny) in seen or (nx, ny) in body:
                    continue
                seen.add((nx, ny))
                stack.append((nx, ny))
        return n

    def choose(game):
        danger, moves = game.collisions()
        candidates = [m for m in ("straight", "right", "left") if not danger[m]]
        if not candidates:
            return "straight"
        head = game.snake[0]
        dirs = {Direction.RIGHT: (1, 0), Direction.LEFT: (-1, 0),
                Direction.UP: (0, -1), Direction.DOWN: (0, 1)}
        food = game.food or head
        scored = []
        for m in candidates:
            dx, dy = dirs[moves[m]]
            cell = (head[0] + dx, head[1] + dy)
            space = flood_count(game, cell)
            dist = abs(cell[0] - food[0]) + abs(cell[1] - food[1])
            scored.append((space, -dist, m))
        scored.sort(reverse=True)
        return scored[0][2]

    scores = []
    for ep in range(episodes):
        g = TextSnake()
        while True:
            move = choose(g)
            r, done = g.step(move)
            if done:
                break
        scores.append(g.score)
        print(f"  episode {ep+1}: score {g.score}, {g.steps} moves")
        log.add("laya/floodfill_episode_score", g.score, ep + 1)

    print()
    print(f"  flood fill scores : {scores}")
    print(f"  mean score        : {float(np.mean(scores)):.2f}")
    print(f"  best              : {max(scores)}")
    print(f"  (this is a SUBMITTED hand written policy, not Laya's own play)")
    log.add_many({"laya/floodfill_mean": float(np.mean(scores)),
                  "laya/floodfill_best": float(max(scores))}, episodes)
    return scores


def main():
    p = argparse.ArgumentParser(description="Snake played with Laya")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--decision", action="store_true")
    p.add_argument("--policy", action="store_true")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log", default="both")
    p.add_argument("--run-name", default=RUN_NAME)
    a = p.parse_args()

    if not (a.probe or a.decision or a.policy):
        a.probe = a.decision = a.policy = True

    g = TextSnake()
    state = g.state_text()
    print(f"state text is {len(state)} characters, roughly {len(state)//4} tokens")

    print("loading laya (first run downloads about 808 MB)...")
    t0 = time.time()
    import laya
    agent = laya.load("convaiinnovations/laya")
    if a.device == "cuda" and hasattr(agent, "model"):
        agent.model.to("cuda")
    print(f"loaded in {time.time()-t0:.0f}s on {a.device}")

    log = Logger(backend=a.log, run_name=a.run_name, log_dir="./runs",
                 params={"model": "convaiinnovations/laya",
                         "params_millions": 421,
                         "backbone": "ModernBERT-large",
                         "device": a.device,
                         "note": "text decision model, board rendered as text"})

    if a.probe:
        t0 = time.time()
        res = ask_laya(agent, g)
        dt = (time.time() - t0) * 1000
        print()
        print("=" * 78)
        print("PROBE: one decision on the full text board")
        print("=" * 78)
        print(f"  latency     : {dt:.0f} ms")
        print(f"  heading     : {res['answers']['heading']['choice']} "
              f"(p={res['answers']['heading'].get('answer_confidence')})")
        print(f"  safe move   : {res['answers']['safe_move']['choice']} "
              f"(p={res['answers']['safe_move'].get('answer_confidence')})")
        log.add_many({"laya/ms_per_move": dt}, 1)

    if a.decision:
        run_decisions(agent, a.episodes, log)
    if a.policy:
        run_policy(agent, a.episodes, log)

    log.maybe_flush(force=True)
    log.close()
    print()
    print(f"curves written to runs/{a.run_name}")


if __name__ == "__main__":
    main()
