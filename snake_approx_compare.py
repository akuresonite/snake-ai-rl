"""Same snake, same loop, four different Q functions.

Why this exists
---------------
The DQN plateaus around 63 to 67 and the state vector is not the limit, so the
next question is whether a different function approximator does better on the
identical problem. This runs the same environment, the same replay buffer, the
same epsilon schedule, the same target rule and the same rewards, and swaps only
the thing that turns a state into Q values.

Approximators
-------------
  dqn        the existing torch network, unchanged, trained every step
  logreg     sklearn LogisticRegression per action
  forest     sklearn RandomForestRegressor
  lightgbm   LGBMRegressor, only if lightgbm is installed

A fairness note that matters, and it is stated in the output too: an environment
step is held constant, but the amount of training per step cannot be. The DQN
takes one gradient step on a batch of 1000 every 8 environment steps. A forest
cannot do that, it refits. So the trees refit less often on larger samples. Both
wall clock and environment steps are reported so the comparison is readable
rather than flattering.

Usage
-----
    python snake_approx_compare.py --models dqn,logreg,forest --steps 400000
    python snake_approx_compare.py --models lightgbm --steps 200000
"""

import argparse
import os
import random
import time
import warnings
from collections import deque

import numpy as np

# sklearn and lightgbm are chatty about things that do not matter here, and the
# spam buries the result table.
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

from snake_core import (
    BATCH_SIZE,
    BLOCK_SIZE,
    Direction,
    EPSILON_DECAY_GAMES,
    EPSILON_MIN,
    EPSILON_START,
    GAMMA,
    MAX_MEMORY,
    REWARD_DEATH,
    REWARD_FOOD,
    STARVATION_FRAMES_PER_SEGMENT,
    set_threads,
)
import snake_ai_training_vec as vec
from snake_logging import Logger, NullLogger


# ==========================================
# Q FUNCTIONS
# ==========================================

class _ConstantModel:
    """Predicts one number. Stands in when a target has a single class."""

    def __init__(self, value):
        self.value = float(value)

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.value, dtype=np.float64)


class QFunction:
    """Turns a batch of states into Q values for the 3 actions."""

    name = "base"
    trains_every = 8          # environment steps between training calls
    trains_on = 1000          # samples per training call

    def q_values(self, states):
        raise NotImplementedError

    def train(self, memory):
        raise NotImplementedError

    def describe(self):
        return self.name


class DQNFunction(QFunction):
    """The existing network and trainer, unchanged."""

    name = "dqn"

    def __init__(self, obs_width, device):
        import torch
        from snake_core import HIDDEN_LAYERS, HIDDEN_SIZE, LR, QTrainer, Linear_QNet
        self.torch = torch
        self.device = device
        self.model = Linear_QNet(obs_width, HIDDEN_SIZE, 3,
                                 hidden_layers=HIDDEN_LAYERS).to(device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=GAMMA)
        self.trains_every = 8
        self.trains_on = BATCH_SIZE
        self.calls = 0
        self.last_fit_seconds = 0.0

    def q_values(self, states):
        torch = self.torch
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return self.model(x).cpu().numpy()

    def train(self, memory):
        if len(memory) < self.trains_on:
            return
        import random
        batch = random.sample(memory, self.trains_on)
        s, a, r, ns, d = zip(*batch)
        self.trainer.train_step(s, a, r, ns, d)
        self.calls += 1

    def describe(self):
        return (f"Linear_QNet {self.model.input_size}->{self.model.hidden_size}"
                f"x{self.model.hidden_layers}->3, AdamW, one batch of {self.trains_on}"
                f" every {self.trains_every} env steps")


class SklearnFunction(QFunction):
    """One sklearn regressor per action, refit on recent experience.

    A separate model per action, because these models predict a single number
    and Q learning needs one value per action. That is the standard way to use
    a non neural Q function.
    """

    name = "sklearn"

    def __init__(self, make_model, refit_every=2000, sample_size=20000, label="sklearn"):
        self.make_model = make_model
        self.label = label
        self.models = None
        self.trains_every = refit_every
        self.trains_on = sample_size
        self.calls = 0
        self.last_fit_seconds = 0.0
        self.sizes_trained = []

    def _fit(self, X, Y):
        models = []
        for a in range(3):
            y = Y[:, a]
            # LogisticRegression refuses a target with one class. That happens
            # for real here: early on, and in dead end states, every sampled
            # transition for an action can carry the same Q value. Keep the
            # previous model for that action instead of crashing the whole run.
            if np.unique(y).size < 2:
                models.append(self.models[a] if self.models else _ConstantModel(float(y[0])))
                continue
            mdl = self.make_model()
            mdl.fit(X, y)
            models.append(mdl)
        return models

    def q_values(self, states):
        X = np.asarray(states)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = X.astype(np.float64)
        if self.models is None:
            return np.zeros((X.shape[0], 3), dtype=np.float32)
        out = np.column_stack([mdl.predict(X) for mdl in self.models])
        return out.astype(np.float32)

    def train(self, memory):
        if len(memory) < min(self.trains_on, 200):
            return
        n = min(self.trains_on, len(memory))
        # A random sample from the whole buffer, the same way the DQN samples
        # its batch. Using only the most recent transitions would bias the tree
        # against old experience, which the DQN does not suffer from.
        batch = random.sample(memory, n)
        S = np.asarray([b[0] for b in batch], dtype=np.float64)
        A = np.asarray([np.argmax(b[1]) for b in batch], dtype=np.int64)
        R = np.asarray([b[2] for b in batch], dtype=np.float64)
        NS = np.asarray([b[3] for b in batch], dtype=np.float64)
        D = np.asarray([b[4] for b in batch], dtype=bool)

        # Same target rule as the DQN trainer, so only the approximator differs.
        next_q = self.q_values(NS)
        target_all = R + GAMMA * next_q.max(axis=1) * (~D)

        Y = self.q_values(S)
        Y[np.arange(len(batch)), A] = target_all[np.arange(len(batch))]

        t0 = time.perf_counter()
        self.models = self._fit(S, Y)
        self.last_fit_seconds = time.perf_counter() - t0
        self.calls += 1
        self.sizes_trained.append(n)

    def describe(self):
        return (f"{self.label}, one model per action, refit on a random "
                f"{self.trains_on} of the replay buffer every "
                f"{self.trains_every} env steps")


# ==========================================
# ENVIRONMENT, identical to the vectorized one
# ==========================================

class Board(vec.Board):
    pass


def remember_many(memory, obs, one_hot, rewards, next_obs, done):
    """Append a whole step of transitions at once.

    Appending one at a time in Python cost about 2 microseconds per game, which
    on 256 games is 0.5 ms per step and turned a 19,000 steps/s loop into a
    1,900 steps/s one.
    """
    for o, a, r, no, d in zip(obs, one_hot, rewards, next_obs, done):
        memory.append((o, a, float(r), no, bool(d)))


def run(model_name, n_games, total_steps, obs_mode, log_backend, run_name):
    import torch
    set_threads(os.environ.get("SNAKE_THREADS") or 1)

    device = torch.device("cpu")   # sklearn is CPU only, so everyone plays here
    env = vec.VecSnake(n_games, obs_mode=obs_mode)
    width = env.obs_width

    if model_name == "dqn":
        qf = DQNFunction(width, device)
    elif model_name == "logreg":
        from sklearn.linear_model import LogisticRegression
        # Plain regression, not classification. The target is a Q value, which
        # is a real number, so a classifier would be the wrong tool.
        qf = SklearnFunction(
            lambda: LogisticRegression(max_iter=300, C=1.0),
            refit_every=2000, sample_size=20000, label="LogisticRegression")
    elif model_name == "forest":
        from sklearn.ensemble import RandomForestRegressor
        # Kept modest on purpose. A big forest spends all its time predicting,
        # and predicting is on the hot path: every refit asks the old forest for
        # Q values on the whole sample, twice.
        qf = SklearnFunction(
            lambda: RandomForestRegressor(n_estimators=30, max_depth=10,
                                          n_jobs=-1, min_samples_leaf=10),
            refit_every=5000, sample_size=10000, label="RandomForestRegressor")
    elif model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            print("=> lightgbm is not installed. Run: pip install lightgbm")
            return None
        qf = SklearnFunction(
            lambda: LGBMRegressor(n_estimators=100, learning_rate=0.1,
                                  num_leaves=31, min_child_samples=20,
                                  verbose=-1, n_jobs=-1),
            refit_every=2000, sample_size=40000, label="LGBMRegressor")
    elif model_name == "tree":
        from sklearn.tree import DecisionTreeRegressor
        qf = SklearnFunction(
            lambda: DecisionTreeRegressor(max_depth=12, min_samples_leaf=10),
            refit_every=5000, sample_size=10000, label="DecisionTreeRegressor")
    elif model_name == "svm":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        # SVR is O(n^2) in the sample count, so it gets a small sample and a
        # long gap between refits. Scaling first matters, the raw state mixes
        # 0/1 flags with distances.
        qf = SklearnFunction(
            lambda: make_pipeline(StandardScaler(),
                                  SVR(kernel="rbf", C=1.0, gamma="scale",
                                      epsilon=0.1)),
            refit_every=10000, sample_size=3000, label="SVR(rbf)")
    elif model_name == "mlp":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPRegressor
        # Small and few iterations on purpose: this refits often, so a big MLP
        # would spend all its time in fit and none in the environment.
        qf = SklearnFunction(
            lambda: make_pipeline(StandardScaler(),
                                  MLPRegressor(hidden_layer_sizes=(64,),
                                               max_iter=40, learning_rate_init=1e-3,
                                               early_stopping=False)),
            refit_every=2000, sample_size=8000, label="MLPRegressor(64)")
    elif model_name == "mlp2":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPRegressor
        qf = SklearnFunction(
            lambda: make_pipeline(StandardScaler(),
                                  MLPRegressor(hidden_layer_sizes=(128, 128),
                                               max_iter=40, learning_rate_init=1e-3,
                                               early_stopping=False)),
            refit_every=2000, sample_size=8000, label="MLPRegressor(128,128)")
    else:
        raise ValueError(f"unknown model {model_name}")

    memory = deque(maxlen=MAX_MEMORY)
    decay_episodes = EPSILON_DECAY_GAMES * (n_games ** 0.5)

    if log_backend == "none":
        log = NullLogger()
    else:
        log = Logger(backend=log_backend, run_name=run_name, log_dir="./runs",
                     params={"model": model_name, "games": n_games,
                             "steps": total_steps, "state": obs_mode,
                             "obs_width": width, "approx": qf.describe()})
        # Say exactly where the curves went, so they are easy to find and
        # compare against the DQN run.
        if getattr(log, "_tb_dir", None):
            print(f"=> tensorboard: {log._tb_dir}")
            print(f"   view with  : tensorboard --logdir ./runs")

    obs = env.observe()
    next_train = qf.trains_every
    steps_done = 0
    t0 = time.perf_counter()
    last_log = t0
    log_step = 0
    games = 0
    record = 0
    best_at_log = 0
    interrupted = False
    fit_seconds = 0.0
    episode_offset = 0

    try:
        while steps_done < total_steps:
            progress = min(1.0, env.episodes / max(1.0, decay_episodes))
            eps = max(EPSILON_MIN, EPSILON_START * ((1.0 - progress) ** 1.5))

            q = qf.q_values(obs)
            explore = np.random.random(n_games) < eps
            actions = q.argmax(axis=1).astype(np.int64)
            actions[explore] = np.random.randint(0, 3, size=int(explore.sum()))

            next_obs, rewards, done, scores = env.step(actions)

            # Vectorized insert, the same helper the production trainer uses.
            # A plain per game Python loop here made the DQN 10 times slower
            # than it is in snake_ai_training_vec.py, which would have poisoned
            # the whole comparison.
            one_hot = np.zeros((n_games, 3), dtype=np.int8)
            one_hot[np.arange(n_games), actions] = 1
            remember_many(memory, obs, one_hot, rewards, next_obs, done)

            # Per episode curve, the same tag the production trainer writes, so
            # the two can be overlaid in TensorBoard.
            for i in np.flatnonzero(done):
                s = int(scores[i])
                games += 1
                log.add("episode/score", s, episode_offset + games)
                if s > record:
                    record = s
                    log.add("episode/best_score", s, episode_offset + games)

            obs = next_obs
            steps_done += n_games

            if steps_done >= next_train:
                before = qf.calls
                qf.train(memory)
                if qf.calls > before:
                    fit_seconds += qf.last_fit_seconds
                next_train += qf.trains_every

            now = time.perf_counter()
            if now - last_log >= 1.0:
                recent = env.episode_scores
                log.add_many({
                    "episode/score_mean_200": float(np.mean(recent)) if recent else 0.0,
                    "episode/score_max_200": float(max(recent)) if recent else 0.0,
                    "episode/best_score": record,
                    "explore/epsilon": eps,
                    "progress/steps": steps_done,
                    "progress/episodes": episode_offset + games,
                    "throughput/steps_per_s": (steps_done - log_step) / max(now - last_log, 1e-9),
                    "train/refits": qf.calls,
                    "train/fit_seconds_total": fit_seconds,
                    "train/fit_seconds_recent": fit_seconds - best_at_log,
                }, steps_done)
                log.maybe_flush()
                last_log = now
                log_step = steps_done
                best_at_log = fit_seconds
    except KeyboardInterrupt:
        interrupted = True

    elapsed = time.perf_counter() - t0
    recent = env.episode_scores
    log.add_many({
        "episode/score_mean_200": float(np.mean(recent)) if recent else 0.0,
        "episode/best_score": record,
        "progress/steps": steps_done,
        "progress/episodes": episode_offset + games,
        "throughput/steps_per_s": steps_done / max(elapsed, 1e-9),
        "train/refits": qf.calls,
        "train/fit_seconds_total": fit_seconds,
    }, steps_done)
    log.maybe_flush(force=True)
    log.close()

    return {
        "model": model_name,
        "features": width,
        "steps": steps_done,
        "games": games,
        "best": record,
        "mean200": float(np.mean(recent)) if recent else 0.0,
        "seconds": elapsed,
        "steps_per_s": steps_done / max(elapsed, 1e-9),
        "fits": qf.calls,
        "fit_seconds": fit_seconds,
        "describe": qf.describe(),
    }


def main():
    p = argparse.ArgumentParser(description="Compare Q function approximators on Snake")
    p.add_argument("--models", default="dqn,logreg,forest",
                   help="comma list of dqn, logreg, forest, lightgbm")
    p.add_argument("--steps", type=int, default=400_000)
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--state", dest="obs_mode", default="basic",
                   choices=sorted(vec.VecSnake.WIDTHS))
    p.add_argument("--seeds", default="0", help="comma list of seeds")
    p.add_argument("--log", dest="log_backend", default="both",
                   help="default both, so TensorBoard always gets the curves",
                   choices=["both", "mlflow", "tensorboard", "none"])
    p.add_argument("--tag", default="approx")
    a = p.parse_args()

    names = [m.strip() for m in a.models.split(",") if m.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    print("=" * 78)
    print(f"approximator comparison | {a.games} games | {a.steps:,} steps | "
          f"state {a.obs_mode} | seeds {seeds}")
    print("=" * 78)
    print("held constant: environment, rewards, replay cap, epsilon schedule,")
    print("               target rule, environment step budget")
    print("NOT constant : training per step. The trees cannot take a gradient")
    print("               step, so they refit less often on larger samples.")
    print("=" * 78)

    rows = []
    for name in names:
        for seed in seeds:
            np.random.seed(seed)
            try:
                import torch
                torch.manual_seed(seed)
            except ImportError:
                pass
            print(f"\n--- {name} seed {seed} ---")
            r = run(name, a.games, a.steps, a.obs_mode, a.log_backend,
                    f"{a.tag}-{name}-s{seed}")
            if r is None:
                continue
            rows.append(r)
            print(f"  best score   {r['best']}")
            print(f"  mean200      {r['mean200']:.1f}")
            print(f"  steps/s      {r['steps_per_s']:,.0f}")
            print(f"  refits       {r['fits']}   fit time {r['fit_seconds']:.1f}s")
            print(f"  {r['describe']}")

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  {'model':10s} {'features':>8} {'best':>5} {'mean200':>8} "
          f"{'steps/s':>9} {'refits':>7} {'fit s':>8}")
    for r in rows:
        print(f"  {r['model']:10s} {r['features']:>8} {r['best']:>5} "
              f"{r['mean200']:>8.1f} {r['steps_per_s']:>9,.0f} "
              f"{r['fits']:>7} {r['fit_seconds']:>8.1f}")

    if rows:
        best = max(rows, key=lambda r: r["best"])
        print()
        print(f"  best score overall: {best['model']} with {best['best']}")


if __name__ == "__main__":
    main()
