"""Teach the learned agent from the search planner.

The finding this is built on
----------------------------
Hand written planning policies on this same board, no learning:

    bfs_safe    mean 132.33   best 187   (BFS to food, only when the tail
                                          stays reachable, else follow the tail)
    flood       mean 123.53   best 176   (take the legal move leaving the most
                                          reachable space)
    greedy      mean  33.93
    random      mean   0.00

Our best learned agent, after 39 million steps: greedy mean 30.9, best 77.

So the ceiling is not the game. It is that we are asking a reactive policy to
beat a planner. The fix is not more training, it is a better signal.

What this does
--------------
The planner plays and writes down what it did. The agent then learns to copy it,
which is imitation learning. The agent still only sees the 11 feature state, so
this measures how much of the planner's ability can be squeezed into a reactive
policy.

Usage
-----
    python tools/imitation.py --episodes 60 --epochs 300
    python tools/imitation.py --episodes 60 --state rich
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch

import snake_core as core
import snake_ai_training_vec as vec
from snake_logging import Logger

# Reuse the planners from the search baseline, so the teacher here is exactly
# the policy that was measured at 132.
from tools.search_baseline import Board, policy_bfs_safe


def collect(n_episodes, obs_mode, seed=0, max_steps=20000):
    """Let the planner play and record (state, action) pairs."""
    np.random.seed(seed)
    env = vec.VecSnake(1, obs_mode=obs_mode, auto_reset=False)
    env.reset()
    X, Y = [], []
    scores = []
    move_names = ["straight", "right", "left"]
    for ep in range(n_episodes):
        env.reset()
        # The planner works in cell space, so mirror the game into a Board.
        b = Board(env.board.cols, env.board.rows)
        b.snake.clear()
        for cell in env.queues[0]:
            b.snake.append((cell[0] // core.BLOCK_SIZE, cell[1] // core.BLOCK_SIZE))
        # queues hold newest first, Board wants newest first too
        b.direction = vec.Direction(env.directions[0])
        b.food = (int(env.foods[0][0]) // core.BLOCK_SIZE,
                  int(env.foods[0][1]) // core.BLOCK_SIZE)
        b.score = int(env.scores[0])
        b.steps = 0

        steps = 0
        while steps < max_steps:
            obs = env.observe()
            teacher_move = policy_bfs_safe(b)
            if teacher_move not in move_names:
                teacher_move = "straight"
            X.append(obs[0].copy())
            Y.append(move_names.index(teacher_move))

            action = np.array([move_names.index(teacher_move)], dtype=np.int64)
            obs, r, done, sc = env.step(action)
            steps += 1

            # Keep the Board mirror in step with the real game.
            b.snake.clear()
            for cell in env.queues[0]:
                b.snake.append((cell[0] // core.BLOCK_SIZE, cell[1] // core.BLOCK_SIZE))
            b.direction = vec.Direction(env.directions[0])
            b.food = (int(env.foods[0][0]) // core.BLOCK_SIZE,
                      int(env.foods[0][1]) // core.BLOCK_SIZE)
            b.score = int(env.scores[0])
            b.steps = steps
            if done[0]:
                break
        scores.append(steps)
    return np.asarray(X), np.asarray(Y), scores


def train_student(X, Y, obs_width, epochs, lr, batch, log, tag):
    torch.manual_seed(0)
    model = core.Linear_QNet(obs_width, core.HIDDEN_SIZE, 3,
                             hidden_layers=core.HIDDEN_LAYERS)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # The teacher turns rarely: measured 92 percent straight, 4 percent right,
    # 4 percent left. Plain cross entropy on that is happy to predict straight
    # forever and still report 92 percent agreement. Weighting the classes by
    # inverse frequency stops the student from learning that shortcut.
    counts = np.bincount(Y, minlength=3).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (3.0 * counts)
    print(f"    class counts {counts.astype(int).tolist()}, "
          f"loss weights {np.round(weights, 2).tolist()}")
    lossf = torch.nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, dtype=torch.float32))

    n = len(X)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    Yt = torch.as_tensor(Y, dtype=torch.long)

    print(f"  training on {n:,} (state, action) pairs for {epochs} epochs")
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        correct = 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xt[idx], Yt[idx]
            pred = model(xb)
            loss = lossf(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            correct += int((pred.argmax(dim=1) == yb).sum())
        acc = correct / n
        log.add("imitation/loss", total / n, ep)
        log.add("imitation/accuracy", acc, ep)
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            print(f"    epoch {ep:>4}  loss {total/n:.4f}  "
                  f"agreement with the teacher {acc*100:.1f} percent")
    return model


def evaluate(model, obs_mode, games=20, seed=99):
    """Play the student greedily and score it, the same way as every other agent."""
    np.random.seed(seed)
    env = vec.VecSnake(1, obs_mode=obs_mode, auto_reset=False)
    scores = []
    for _ in range(games):
        env.reset()
        while True:
            obs = env.observe()
            with torch.no_grad():
                q = model(torch.as_tensor(obs, dtype=torch.float32)).numpy()
            a = q.argmax(axis=1).astype(np.int64)
            obs, r, done, sc = env.step(a)
            if done[0]:
                scores.append(int(sc[0]))
                break
    return scores


def evaluate_always_straight(obs_mode, games=20, seed=99):
    """The lazy policy the class imbalance would reward. Needs a real baseline.

    With 92 percent of the teacher's actions being "straight", a student that
    only ever goes straight reports high agreement while playing badly. Measuring
    this makes that visible instead of letting it look like success.
    """
    np.random.seed(seed)
    env = vec.VecSnake(1, obs_mode=obs_mode, auto_reset=False)
    scores = []
    for _ in range(games):
        env.reset()
        while True:
            a = np.zeros(1, dtype=np.int64)   # 0 is straight
            obs, r, done, sc = env.step(a)
            if done[0]:
                scores.append(int(sc[0]))
                break
    return scores


def main():
    p = argparse.ArgumentParser(description="Imitation learning from a search planner")
    p.add_argument("--episodes", type=int, default=60,
                   help="teacher games to record")
    p.add_argument("--state", dest="obs_mode", default="basic",
                   choices=sorted(vec.VecSnake.WIDTHS))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--log", default="both")
    p.add_argument("--run-name", default=None)
    a = p.parse_args()

    run_name = a.run_name or f"s95__imitation-{a.obs_mode}"
    log = Logger(backend=a.log, run_name=run_name, log_dir=os.path.join(ROOT, "runs"),
                 params={"teacher": "policy_bfs_safe", "teacher_mean_score": 132.33,
                         "state": a.obs_mode, "episodes": a.episodes,
                         "epochs": a.epochs, "lr": a.lr})

    print("=" * 84)
    print("IMITATION LEARNING, student copies the BFS planner")
    print("=" * 84)
    print(f"  teacher: BFS to the food, only when the tail stays reachable")
    print(f"  teacher measured at mean 132.33, best 187, no learning")
    print()

    t0 = time.perf_counter()
    print(f"  recording {a.episodes} teacher games...")
    X, Y, lens = collect(a.episodes, a.obs_mode)
    print(f"    {len(X):,} pairs in {time.perf_counter()-t0:.0f}s, "
          f"mean game length {np.mean(lens):.0f} moves")
    print(f"    action balance: straight {np.mean(Y==0)*100:.0f} percent, "
          f"right {np.mean(Y==1)*100:.0f} percent, "
          f"left {np.mean(Y==2)*100:.0f} percent")

    model = train_student(X, Y, X.shape[1], a.epochs, a.lr, a.batch, log, a.obs_mode)

    print()
    print("  evaluating the student greedily...")
    scores = evaluate(model, a.obs_mode, games=a.eval_games)
    lazy = evaluate_always_straight(a.obs_mode, games=a.eval_games, seed=99)
    print(f"    student        : mean {np.mean(scores):.2f}, best {max(scores)}")
    print(f"    always straight: mean {np.mean(lazy):.2f}, best {max(lazy)}")
    if np.mean(scores) <= np.mean(lazy):
        print("    WARNING: the student is no better than always going straight.")
        print("    High teacher agreement on an unbalanced dataset does not mean")
        print("    it learned to play. Read this result with that in mind.")
    log.add_many({"imitation/student_mean": float(np.mean(scores)),
                  "imitation/student_best": float(max(scores)),
                  "imitation/always_straight_mean": float(np.mean(lazy)),
                  "imitation/teacher_mean": 132.33}, 0)

    print()
    print("=" * 84)
    print("WHERE THIS SITS")
    print("=" * 84)
    print(f"  {'agent':38s} {'mean':>8} {'best':>6}")
    print(f"  {'random':38s} {0.00:>8.2f} {0:>6}")
    print(f"  {'greedy hand policy':38s} {33.93:>8.2f} {71:>6}")
    print(f"  {'DQN 1.2M steps':38s} {32.40:>8.2f} {60:>6}")
    print(f"  {'your record 77 checkpoint':38s} {30.90:>8.2f} {67:>6}")
    print(f"  {'student, imitation':38s} {np.mean(scores):>8.2f} {max(scores):>6}")
    print(f"  {'flood planner, needs full board':38s} {123.53:>8.2f} {176:>6}")
    print(f"  {'bfs_safe planner, needs full board':38s} {132.33:>8.2f} {187:>6}")

    pth = os.path.join(ROOT, "model", f"{run_name}.pth")
    torch.save({"model_state": model.state_dict(),
                "architecture": core.architecture_fingerprint(input_size=X.shape[1]),
                "n_games": 0, "record": int(max(scores)), "total_score": 0,
                "stopped_early": False,
                "note": "imitation student of policy_bfs_safe"},
               pth)
    print()
    print(f"  saved {os.path.relpath(pth, ROOT)}")
    log.maybe_flush(force=True)
    log.close()


if __name__ == "__main__":
    main()
