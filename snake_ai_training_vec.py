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
# BOARD GEOMETRY, no pygame
# ==========================================

class Board:
    """Just the rules. No window, no clock, no drawing.

    The single game class carries pygame calls through every method, which makes
    it impossible to run several of them in a tight loop. This class is the same
    rules with the rendering stripped out.
    """

    __slots__ = ("cols", "rows", "game_w", "game_h")

    def __init__(self, w=800, h=600, block=BLOCK_SIZE):
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
    """

    def __init__(self, n_games, w=800, h=600, auto_reset=True):
        self.n = n_games
        self.board = Board(w, h)
        self.auto_reset = auto_reset
        self.steps = 0
        self.episodes = 0
        self.episode_returns = deque(maxlen=200)
        self.episode_scores = deque(maxlen=200)
        self.episode_lengths = deque(maxlen=200)
        self.episode_best = 0
        self._returns = np.zeros(n_games, dtype=np.float32)
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
        """Stack the 11 value state for every game. Returns (n, 11) int8."""
        n = self.n
        out = np.zeros((n, 11), dtype=np.int8)

        # Danger straight, right, left for each direction. Same table as the
        # single game version, applied to all games at once.
        for i in range(n):
            hx, hy = int(self.heads[i, 0]), int(self.heads[i, 1])
            d = self.directions[i]
            left = (hx - BLOCK_SIZE, hy)
            right = (hx + BLOCK_SIZE, hy)
            up = (hx, hy - BLOCK_SIZE)
            down = (hx, hy + BLOCK_SIZE)

            if d == Direction.RIGHT.value:
                straight, r_turn, l_turn = right, down, up
            elif d == Direction.DOWN.value:
                straight, r_turn, l_turn = down, left, up
            elif d == Direction.LEFT.value:
                straight, r_turn, l_turn = left, up, down
            else:  # UP
                straight, r_turn, l_turn = up, right, left

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
            hx, hy = int(self.heads[i, 0]), int(self.heads[i, 1])
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


# ==========================================
# PART 2: AGENT (shared network and replay)
# ==========================================

class VecAgent:
    """One network, one replay buffer, N games feeding it."""

    def __init__(self, device):
        self.device = device
        self.n_games = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(
            input_size=11,
            hidden_size=HIDDEN_SIZE,
            output_size=3,
            hidden_layers=HIDDEN_LAYERS,
        ).to(device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=GAMMA)
        self.record = 0
        self.total_score = 0
        self.decay_episodes = float(EPSILON_DECAY_GAMES)

    def architecture(self):
        # Shared with the single game script, so a checkpoint written by either
        # one is recognised by the other.
        return architecture_fingerprint()

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


def train(n_games, total_steps, train_every, device_name, checkpoint,
          best_checkpoint=VEC_BEST_CHECKPOINT, save_every=60.0, quiet=False,
          log_backend="both", run_name=None, log_dir="./runs", report_every=5.0,
          log_every=1.0):
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
    """
    device = resolve_device(device_name)
    print(f"=> {n_games} parallel games on {device}")
    if device.type == "cuda":
        print(f"=> GPU: {torch.cuda.get_device_name(0)}")

    env = VecSnake(n_games)
    agent = VecAgent(device)
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
    report_step = steps_done
    log_step = steps_done
    train_calls = 0
    diag_acc = {}
    interrupted = False

    try:
        while steps_done < total_steps:
            eps = agent.epsilon_for(env.episodes)
            actions = agent.act(obs, eps)

            next_obs, rewards, done, scores = env.step(actions)

            # next_obs is the post reset observation for finished games. That is
            # fine here because done=True makes the bootstrap term fall away in
            # the trainer, so the next state is never actually read.
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

            if steps_done >= next_train:
                agent.train_on_replay()
                train_calls += 1
                next_train += train_every
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
                agent.save(checkpoint, extra={"steps": total_seen})
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
        agent.save(checkpoint, extra={"steps": total_seen})
        print(f"=> Saved {checkpoint}")

    elapsed = time.perf_counter() - t0
    # Always leave a current latest file behind, whether we finished the run or
    # the user stopped it.
    agent.save(checkpoint, extra={"steps": total_seen})

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
    p.add_argument("--steps", type=int, default=500_000, help="total environment steps")
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
    p.add_argument("--quiet", action="store_true",
                   help="silence the terminal, logging still happens")
    a = p.parse_args()
    train(a.games, a.steps, a.train_every, a.device, a.checkpoint,
          a.best_checkpoint, a.save_every, a.quiet, a.log_backend,
          a.run_name, a.log_dir, a.report_every, a.log_every)


if __name__ == "__main__":
    main()
