"""Every ML algorithm copies the search planner, with the comparison in TensorBoard.

Why a saved dataset
-------------------
The flood state costs about 75 observations per second for a single game against
94,610 environment steps per second for the basic state, because flood fill runs
for every game every step. Recording the teacher once and caching it means that
cost is paid ONCE, not once per algorithm. Every algorithm then sees exactly the
same rows, which is what makes the comparison fair.

What is compared
----------------
  state:  basic, 11 features   or   flood, 25 features with trap geometry
  algo:   logreg, tree, forest, lightgbm, svm, mlp, mlp2, dqn

All of them are trained on the same (state, action) pairs from the BFS planner,
which scores mean 132 with no learning, and evaluated on the same fresh games,
greedy and without exploration.

How a non neural model becomes a policy
---------------------------------------
Every algorithm is used as a CLASSIFIER: one binary model per action, answering
"did the teacher take this action in this state?". The action with the highest
probability is played. The DQN does the same thing with a softmax over 3
outputs, so this is the closest non neural equivalent.

Two earlier designs were wrong, and both looked fine in the logs:

1. Trained on a label with no action column. Result: 90 percent agreement with
   the teacher and a score of 0, because nothing let the model COMPARE actions.
2. Trained to REGRESS the discounted return. That looks more principled, and it
   is what a Q function actually is, but it fails here: most transitions return
   0, only death is -10 and food is +10, so the target is sparse and every model
   predicts close to the mean for all three actions. Argmax over three nearly
   equal numbers is noise. Agreement stayed near 90 percent while the score
   stayed at 0.

The lesson: teacher agreement is not the same as playing well, and it must never
be reported on its own.

Usage
-----
    python tools/ml_imitation.py --episodes 15 --state basic --algo all
    python tools/ml_imitation.py --episodes 10 --state flood --algo all --stride 4
    python tools/ml_imitation.py --list
"""

import argparse
import os
import pickle
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import warnings

# sklearn and joblib are extremely chatty here, and the spam hides the results.
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
import torch

import snake_core as core
import snake_ai_training_vec as vec
from snake_logging import Logger
from tools.search_baseline import Board, policy_bfs_safe

CACHE_DIR = os.path.join(ROOT, "model", "teacher_data")
MOVE_NAMES = ["straight", "right", "left"]
TEACHER_MEAN = 132.33
TEACHER_BEST = 187
ALGOS = ["logreg", "tree", "forest", "lightgbm", "svm", "mlp", "mlp2", "dqn"]


class _ConstModel:
    """Predicts a constant. Stands in when a target has only one class."""

    classes_ = np.array([0, 1])

    def __init__(self, value):
        self.value = float(value)

    def predict_proba(self, X):
        n = np.asarray(X).shape[0]
        p = np.zeros((n, 2))
        p[:, 1] = self.value
        p[:, 0] = 1.0 - self.value
        return p

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.value)


# ==========================================
# RECORDING THE TEACHER
# ==========================================

def record(episodes, obs_mode, seed=0, max_steps=20000, stride=1):
    """Let the planner play and record (state, action, return) triples.

    stride records only every Nth step. That is the real fix for the flood state
    cost: micro optimising the array wrapper made it SLOWER, so the answer is to
    ask for the expensive thing less often. The teacher's moves are highly
    correlated between adjacent steps, so a stride of 4 loses little.
    """
    path = os.path.join(CACHE_DIR,
                        f"teacher_{obs_mode}_{episodes}_{seed}_s{stride}.pkl")
    if os.path.exists(path):
        print(f"  using cached {os.path.relpath(path, ROOT)}")
        with open(path, "rb") as f:
            return pickle.load(f)

    print(f"  recording {episodes} teacher games in {obs_mode} state, "
          f"every {stride} step")
    if obs_mode == "flood":
        print(f"    flood fill is the slow part, roughly "
              f"{episodes * 5000 * stride / 75 / 60:.0f} minutes")
    np.random.seed(seed)
    env = vec.VecSnake(1, obs_mode=obs_mode, auto_reset=False)
    X, Y, R = [], [], []
    game_scores = []
    t0 = time.perf_counter()

    for ep in range(episodes):
        env.reset()
        b = Board(env.board.cols, env.board.rows)
        ep_states, ep_rewards, ep_actions = [], [], []
        steps = 0
        while steps < max_steps:
            b.snake.clear()
            for cell in env.queues[0]:
                b.snake.append((cell[0] // core.BLOCK_SIZE, cell[1] // core.BLOCK_SIZE))
            b.direction = vec.Direction(env.directions[0])
            b.food = (int(env.foods[0][0]) // core.BLOCK_SIZE,
                      int(env.foods[0][1]) // core.BLOCK_SIZE)
            b.score = int(env.scores[0])
            b.steps = steps

            move = policy_bfs_safe(b)
            if move not in MOVE_NAMES:
                move = "straight"

            if steps % stride == 0:
                obs = env.observe()
                ep_states.append(obs[0].copy())
                ep_rewards.append(0.0)
                ep_actions.append(MOVE_NAMES.index(move))

            a = np.array([MOVE_NAMES.index(move)], dtype=np.int64)
            obs, r, done, sc = env.step(a)
            if ep_rewards:
                ep_rewards[-1] = float(r)
            steps += 1
            if done[0]:
                # Discounted return per state, kept as a sample weight by the
                # trainer. See the module docstring for why it is NOT the
                # regression target.
                g = 0.0
                rets = [0.0] * len(ep_rewards)
                for i in range(len(ep_rewards) - 1, -1, -1):
                    g = ep_rewards[i] + core.GAMMA * g
                    rets[i] = g
                X.extend(ep_states)
                Y.extend(ep_actions)
                R.extend(rets)
                break
        game_scores.append(int(env.scores[0]))
        if (ep + 1) % 5 == 0 or ep == 0:
            rate = (ep + 1) / max(time.perf_counter() - t0, 1e-9)
            print(f"    game {ep+1}/{episodes}, score {game_scores[-1]}, "
                  f"{len(X):,} pairs, {rate:.2f} games per second")

    data = {"X": np.asarray(X), "Y": np.asarray(Y),
            "R": np.asarray(R, dtype=np.float32), "scores": game_scores,
            "obs_mode": obs_mode, "stride": stride}
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  cached to {os.path.relpath(path, ROOT)}")
    print(f"  teacher scored mean {np.mean(game_scores):.1f}, "
          f"best {max(game_scores)} over {episodes} games")
    return data


# ==========================================
# THE ALGORITHMS
# ==========================================

def build(algo):
    """Return (model, kind). kind is one of logreg, dqn, clf."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if algo == "logreg":
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=500, C=1.0)), "logreg"
    if algo == "tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=14, min_samples_leaf=5,
                                      class_weight="balanced"), "clf"
    if algo == "forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=60, max_depth=16,
                                      min_samples_leaf=5, n_jobs=1,
                                      class_weight="balanced"), "clf"
    if algo == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=150, learning_rate=0.1, num_leaves=31,
                              min_child_samples=20, verbose=-1, n_jobs=1,
                              class_weight="balanced"), "clf"
    if algo == "svm":
        from sklearn.svm import SVC
        return make_pipeline(StandardScaler(),
                             SVC(kernel="rbf", C=1.0, gamma="scale",
                                 probability=True, class_weight="balanced")), "clf"
    if algo == "mlp":
        from sklearn.neural_network import MLPClassifier
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64,), max_iter=200,
                                           learning_rate_init=1e-3)), "clf"
    if algo == "mlp2":
        from sklearn.neural_network import MLPClassifier
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(128, 128),
                                           max_iter=200,
                                           learning_rate_init=1e-3)), "clf"
    if algo == "dqn":
        return None, "dqn"
    raise ValueError(f"unknown algo {algo}")


def action_scores(models, kind, X):
    """(n, 3) scores, one per action, so argmax gives the action."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n = X.shape[0]
    out = np.zeros((n, 3))
    for a in range(3):
        mdl = models[a]
        proba = mdl.predict_proba(X)
        classes = list(getattr(mdl, "classes_", [0, 1]))
        idx = classes.index(1) if 1 in classes else (len(classes) - 1)
        out[:, a] = proba[:, idx]
    return out


def _weight_key(mdl):
    """sample_weight is not accepted through a Pipeline, so say so."""
    name = type(mdl).__name__
    if "Pipeline" in name:
        return None
    return "sample_weight"


def train_algo(algo, X, Y, R, log, epochs=300):
    """Train one algorithm to choose the teacher's action."""
    n, width = X.shape
    counts = np.bincount(Y, minlength=3).astype(float)
    counts[counts == 0] = 1.0
    print(f"    class balance straight/right/left: {counts.astype(int).tolist()}")
    print(f"    returns mean {R.mean():.2f}, min {R.min():.2f}, max {R.max():.2f}")

    r = np.asarray(R, dtype=np.float64)
    lo, hi = float(r.min()), float(r.max())
    w = np.ones(n) if hi <= lo else 0.5 + 1.5 * (r - lo) / (hi - lo)
    print(f"    sample weights min {w.min():.2f}, max {w.max():.2f}")

    _obj, kind = build(algo)

    if kind == "dqn":
        torch.manual_seed(0)
        net = core.Linear_QNet(width, core.HIDDEN_SIZE, 3,
                               hidden_layers=core.HIDDEN_LAYERS)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        cw = counts.sum() / (3.0 * counts)
        lossf = torch.nn.CrossEntropyLoss(
            weight=torch.as_tensor(cw, dtype=torch.float32), reduction="none")
        Xt = torch.as_tensor(X, dtype=torch.float32)
        Yt = torch.as_tensor(Y, dtype=torch.long)
        Wt = torch.as_tensor(w, dtype=torch.float32)
        batch = 256
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot, corr = 0.0, 0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                pred = net(Xt[idx])
                loss = (lossf(pred, Yt[idx]) * Wt[idx]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(idx)
                corr += int((pred.argmax(dim=1) == Yt[idx]).sum())
            log.add("imitation/loss", tot / n, ep)
            log.add("imitation/accuracy", corr / n, ep)
            if ep % max(1, epochs // 8) == 0 or ep == epochs - 1:
                print(f"      epoch {ep:>4}  loss {tot/n:.4f}  "
                      f"agreement {corr/n*100:.1f} percent")
        return ("dqn", net)

    t0 = time.perf_counter()
    models = []
    for a in range(3):
        mdl, _mk = build(algo)
        y = (Y == a).astype(int)
        if len(np.unique(y)) < 2:
            models.append(_ConstModel(float(y[0])))
            continue
        Xf = X.astype(np.float64)
        key = _weight_key(mdl)
        if key is None:
            # A pipeline, so no sample_weight path. Fitted unweighted. Stated
            # rather than hidden, because it makes those arms not quite equal.
            mdl.fit(Xf, y)
        else:
            mdl.fit(Xf, y, **{key: w})
        models.append(mdl)
    fit_s = time.perf_counter() - t0

    scores = action_scores(models, kind, X)
    agree = float((scores.argmax(axis=1) == Y).mean())
    log.add("imitation/accuracy", agree, 0)
    log.add("imitation/fit_seconds", fit_s, 0)
    print(f"    fit in {fit_s:.1f}s, agreement with the teacher {agree*100:.1f} percent")
    return (kind, models)


def play(kind, model_obj, obs_mode, games=20, seed=99):
    np.random.seed(seed)
    env = vec.VecSnake(1, obs_mode=obs_mode, auto_reset=False)
    scores = []
    for _ in range(games):
        env.reset()
        while True:
            obs = env.observe()
            if kind == "dqn":
                with torch.no_grad():
                    q = model_obj(torch.as_tensor(obs, dtype=torch.float32)).numpy()
                a = q.argmax(axis=1).astype(np.int64)
            else:
                s = action_scores(model_obj, kind, obs)
                a = s.argmax(axis=1).astype(np.int64)
            obs, r, done, sc = env.step(a)
            if done[0]:
                scores.append(int(sc[0]))
                break
    return scores


# ==========================================

def main():
    p = argparse.ArgumentParser(description="ML algorithms copy the search planner")
    p.add_argument("--episodes", type=int, default=15)
    p.add_argument("--state", default="basic", choices=["basic", "flood", "both"])
    p.add_argument("--algo", default="all")
    p.add_argument("--epochs", type=int, default=300, help="for the DQN only")
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--stride", type=int, default=1,
                   help="record every Nth step. Use 4 for the flood state")
    p.add_argument("--log", default="both")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()

    states = ["basic", "flood"] if a.state == "both" else [a.state]
    algos = ALGOS if a.algo == "all" else [x.strip() for x in a.algo.split(",")]

    if a.list:
        print("states:", states)
        print("algos :", algos)
        return

    print("=" * 92)
    print(f"EVERY ML ALGORITHM COPIES THE PLANNER | states {states} | "
          f"{len(algos)} algos | stride {a.stride}")
    print("=" * 92)
    print(f"  teacher  : bfs_safe, mean {TEACHER_MEAN}, best {TEACHER_BEST}, no learning")
    print(f"  control  : always straight scores mean 0.00")
    print()

    results = []
    for obs_mode in states:
        data = record(a.episodes, obs_mode, stride=a.stride)
        X, Y, R = data["X"], data["Y"], data["R"]
        print(f"  {obs_mode}: {len(X):,} pairs, {X.shape[1]} features")
        for algo in algos:
            run_name = f"s96__{obs_mode}-{algo}"
            print(f"  --- {run_name} ---")
            log = Logger(backend=a.log, run_name=run_name,
                         log_dir=os.path.join(ROOT, "runs"),
                         params={"teacher": "bfs_safe", "state": obs_mode,
                                 "algo": algo, "features": int(X.shape[1]),
                                 "pairs": int(len(X)), "stride": a.stride,
                                 "teacher_mean": TEACHER_MEAN})
            try:
                t0 = time.perf_counter()
                kind, model_obj = train_algo(algo, X, Y, R, log, epochs=a.epochs)
                total = time.perf_counter() - t0
                scores = play(kind, model_obj, obs_mode, games=a.eval_games)
                mean, best = float(np.mean(scores)), int(max(scores))
                print(f"    plays: mean {mean:.2f}, best {best}, {total:.1f}s total")
                log.add_many({"imitation/play_mean": mean,
                              "imitation/play_best": best,
                              "imitation/total_seconds": total,
                              "imitation/teacher_mean": TEACHER_MEAN}, 0)
                results.append((obs_mode, algo, mean, best, total))
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}")
                results.append((obs_mode, algo, float("nan"), 0, 0.0))
            finally:
                log.maybe_flush(force=True)
                log.close()

    print()
    print("=" * 92)
    print("COMPARISON, same teacher, same rows, same test games")
    print("=" * 92)
    print(f"  {'state':7s} {'algo':10s} {'play mean':>10} {'play best':>10} {'seconds':>9}")
    ok = [r for r in results if r[2] == r[2]]
    for obs_mode, algo, mean, best, secs in sorted(ok, key=lambda x: -x[2]):
        print(f"  {obs_mode:7s} {algo:10s} {mean:>10.2f} {best:>10} {secs:>9.1f}")
    for obs_mode, algo, mean, best, secs in results:
        if mean != mean:
            print(f"  {obs_mode:7s} {algo:10s} {'FAILED':>10} {best:>10} {secs:>9.1f}")
    print()
    print(f"  teacher, no learning    : mean {TEACHER_MEAN:>7.2f}  best {TEACHER_BEST}")
    print(f"  always straight control : mean {0.0:>7.2f}  best {0}")
    if ok:
        top = max(ok, key=lambda r: r[2])
        print(f"  best learned            : {top[0]} with {top[1]}, "
              f"mean {top[2]:.2f}, best {top[3]}")
    print()
    print("  curves are in TensorBoard under s96__<state>-<algo>")


if __name__ == "__main__":
    main()
