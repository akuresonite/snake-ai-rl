"""Score every sweep checkpoint the same way, so the configurations compare.

Why this is needed
------------------
Early stopping means each configuration stopped at a different step count, and
the record inside a checkpoint is the best score seen during training, which was
measured while the policy was still exploring. Two runs with different budgets
are therefore not directly comparable from their own records.

This evaluates every checkpoint the same way: greedy play, no exploration, N
fresh games, same seed. That gives each configuration the same test.

Also reports what a trivial policy scores, so the numbers have a floor to sit on.

Usage
-----
    python tools/eval_sweep.py --games 40
    python tools/eval_sweep.py --games 40 --dir model/sweep
"""

import argparse
import glob
import os
import sys
from collections import deque

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


def load_agent(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck.get("architecture", {})
    width = int(arch.get("input_size", 11))
    obs_mode = "basic" if width == 11 else "rich"
    model = core.Linear_QNet(width, core.HIDDEN_SIZE, 3,
                             hidden_layers=core.HIDDEN_LAYERS,
                             use_norm=arch.get("use_norm", True)).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, obs_mode, width, ck


def greedy_play(model, obs_mode, episodes, seed=12345):
    """Play with no exploration at all and return the scores."""
    np.random.seed(seed)
    env = vec.VecSnake(1, obs_mode=obs_mode)
    # One game at a time so each episode is independent.
    scores = []
    for _ in range(episodes):
        # Reset, then step until done, recording the score at the end.
        obs = env.observe()
        while True:
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32)
                q = model(x).numpy()
            action = q.argmax(axis=1).astype(np.int64)
            obs, r, done, sc = env.step(action)
            if done[0]:
                scores.append(int(sc[0]))
                break
    return scores


def blind_policies(episodes, seed=999):
    """Baselines with no learning, so the numbers have a floor."""
    np.random.seed(seed)
    out = {}
    for label, policy in (("random", None), ("straight", "straight")):
        env = vec.VecSnake(1, obs_mode="basic")
        scores = []
        for _ in range(episodes):
            obs = env.observe()
            while True:
                if policy is None:
                    a = np.array([np.random.randint(0, 3)])
                else:
                    a = np.array([0])
                obs, r, done, sc = env.step(a)
                if done[0]:
                    scores.append(int(sc[0]))
                    break
        out[label] = scores
    return out


def main():
    p = argparse.ArgumentParser(description="Evaluate sweep checkpoints fairly")
    p.add_argument("--dir", default=os.path.join(ROOT, "model", "sweep"))
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--pattern", default="*_best.pth")
    a = p.parse_args()

    paths = sorted(glob.glob(os.path.join(a.dir, a.pattern)))
    if not paths:
        print(f"no checkpoints matching {a.pattern} in {a.dir}")
        return

    print("=" * 92)
    print(f"FAIR EVALUATION | {a.games} greedy games each | no exploration")
    print("=" * 92)

    floors = blind_policies(a.games)
    print("floor, no learning at all:")
    for label, s in floors.items():
        print(f"  {label:10s} mean {np.mean(s):6.2f}  best {max(s):>3}")

    print()
    print(f"  {'checkpoint':24s} {'trained':>8} {'mean':>7} {'median':>7} "
          f"{'best':>5} {'>=10':>6} {'>=40':>6}")
    rows = []
    for path in paths:
        name = os.path.basename(path).replace("_best.pth", "")
        try:
            model, obs_mode, width, ck = load_agent(path)
            scores = greedy_play(model, obs_mode, a.games)
        except Exception as e:
            print(f"  {name:24s} FAILED {type(e).__name__}: {e}")
            continue
        rows.append((name, ck.get("record"), np.mean(scores), np.median(scores),
                     max(scores), sum(1 for x in scores if x >= 10),
                     sum(1 for x in scores if x >= 40), scores))
        r = rows[-1]
        print(f"  {name:24s} {str(r[1]):>8} {r[2]:>7.2f} {r[3]:>7.1f} "
              f"{r[4]:>5} {r[5]:>6} {r[6]:>6}")

    if rows:
        print()
        print("=" * 92)
        print("RANKED BY GREEDY MEAN SCORE")
        print("=" * 92)
        for r in sorted(rows, key=lambda x: -x[2]):
            print(f"  {r[0]:24s} mean {r[2]:6.2f}  best {r[4]:>3}   "
                  f"training record {r[1]}")
        best = max(rows, key=lambda x: x[2])
        print()
        print(f"  winner: {best[0]} with greedy mean {best[2]:.2f} "
              f"and best {best[4]}")


if __name__ == "__main__":
    main()
