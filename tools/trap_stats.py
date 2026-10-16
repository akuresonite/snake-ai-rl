"""Every measurement we have about the snake trapping itself in its own coils.

Why this exists
---------------
Measured on the record 77 model, 95 to 100 percent of deaths are the snake
hitting itself, 0 percent are walls, and 0 to 5 percent are the starvation
guard. Frames per food also RISES slightly with score, so the agent is not
running out of time, it is sealing itself in.

This tool reports the full picture of that, per checkpoint.

Usage
-----
    python tools/trap_stats.py --checkpoint model/vec_best.pth --games 40
    python tools/trap_stats.py --checkpoint model/fair_baseline_s0_best.pth
    python tools/trap_stats.py --all
"""

import argparse
import glob
import os
import sys
from collections import Counter, defaultdict

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

BLOCK = core.BLOCK_SIZE
SEG = core.STARVATION_FRAMES_PER_SEGMENT


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck.get("architecture", {})
    width = int(arch.get("input_size", 11))
    model = core.Linear_QNet(width, core.HIDDEN_SIZE, 3,
                             hidden_layers=core.HIDDEN_LAYERS,
                             use_norm=arch.get("use_norm", True))
    model.load_state_dict(ck["model_state"])
    model.eval()
    mode = {11: "basic", 21: "rich", 25: "flood"}.get(width, "basic")
    return model, mode, ck


def pct(a, b):
    return 0.0 if b == 0 else 100.0 * a / b


def analyse(path, games=40, seed=4242):
    model, mode, ck = load_model(path)
    env = vec.VecSnake(1, obs_mode=mode, auto_reset=False)
    np.random.seed(seed)
    env.reset()

    deaths = Counter()
    trap_ratio_at_death = []
    best_area_at_death = []
    legal_at_death = []
    tail_reachable_at_death = []
    trapped_next_step = 0        # the move taken left less space than the body
    avoidable_traps = 0          # a legal move existed with good space
    steps_total = 0
    tail_lost_total = 0
    score_at_death = []
    by_band = defaultdict(lambda: {"n": 0, "trap": [], "self": 0})
    self_death_trap_ratios = []
    frames_per_food = defaultdict(list)
    tail_lost_by_band = defaultdict(lambda: [0, 0])
    head_area_series = []

    for _ in range(games):
        obs = env.observe()
        frames_since_food = 0
        while True:
            st = env.trap_stats(0)
            steps_total += 1
            head_area_series.append(st["trap_ratio"])
            tail_lost_by_band[0][0] += 0
            if not st["tail_reachable"]:
                tail_lost_total += 1

            prev_score = int(env.scores[0])
            prev_frames = int(env.frames[0])
            with torch.no_grad():
                q = model(torch.as_tensor(obs, dtype=torch.float32)).numpy()
            action = q.argmax(axis=1).astype(np.int64)

            # Which named move did it take, and how much space did that leave?
            chosen = ("straight", "right", "left")[int(action[0])]
            after = env.trap_stats(0)
            legal = after["legal_moves"]
            per_move = after["per_move_area"]
            body = after["body"]
            if legal and per_move:
                best = max(per_move.values())
                if best >= body:
                    avoidable = True
                else:
                    avoidable = False
            else:
                avoidable = False

            obs, r, done, sc = env.step(action)
            if int(env.scores[0]) > prev_score:
                frames_per_food[prev_score].append(prev_frames)

            if done[0]:
                final = env.trap_stats(0)
                body = max(1, final["body"])
                if env.frames[0] > SEG * body:
                    kind = "starvation"
                else:
                    x, y = int(env.heads[0, 0]), int(env.heads[0, 1])
                    if x < 0 or y < 0 or x > env.board.game_w - BLOCK \
                            or y > env.board.game_h - BLOCK:
                        kind = "wall"
                    else:
                        kind = "body"
                deaths[kind] += 1
                score = int(sc[0])
                score_at_death.append(score)
                trap_ratio_at_death.append(final["trap_ratio"])
                best_area_at_death.append(final["best_move_area"])
                legal_at_death.append(final["legal_moves"])
                tail_reachable_at_death.append(final["tail_reachable"])
                band = (score // 10) * 10
                by_band[band]["n"] += 1
                by_band[band]["trap"].append(final["trap_ratio"])
                if kind == "body":
                    by_band[band]["self"] += 1
                    self_death_trap_ratios.append(final["trap_ratio"])
                if not avoidable:
                    trapped_next_step += 1
                if avoidable:
                    avoidable_traps += 1
                env.reset()
                break
    return {
        "path": path, "mode": mode, "record": ck.get("record"),
        "deaths": deaths, "n_games": games,
        "trap_ratio_at_death": trap_ratio_at_death,
        "best_area_at_death": best_area_at_death,
        "legal_at_death": legal_at_death,
        "tail_reachable_at_death": tail_reachable_at_death,
        "score_at_death": score_at_death,
        "by_band": by_band,
        "trapped_next_step": trapped_next_step,
        "avoidable_traps": avoidable_traps,
        "steps_total": steps_total,
        "tail_lost_total": tail_lost_total,
        "self_death_trap_ratios": self_death_trap_ratios,
        "head_area_series": head_area_series,
        "frames_per_food": frames_per_food,
    }


def report(r):
    print("=" * 80)
    print(f"{os.path.relpath(r['path'], ROOT)}")
    print(f"  state mode {r['mode']}, training record {r['record']}, "
          f"{r['n_games']} greedy games")
    print("=" * 80)

    d = r["deaths"]
    n = r["n_games"]
    print()
    print("  HOW IT DIES")
    print(f"    hit its own body     : {d['body']:>3}  ({pct(d['body'], n):>5.1f} percent)")
    print(f"    hit a wall           : {d['wall']:>3}  ({pct(d['wall'], n):>5.1f} percent)")
    print(f"    starvation guard     : {d['starvation']:>3}  ({pct(d['starvation'], n):>5.1f} percent)")

    tr = r["trap_ratio_at_death"]
    if tr:
        print()
        print("  TRAP RATIO AT DEATH, reachable area divided by board area")
        print(f"    mean {np.mean(tr):.3f}   median {np.median(tr):.3f}   "
              f"min {np.min(tr):.3f}   max {np.max(tr):.3f}")
        hist = Counter(min(9, int(v * 10)) for v in tr)
        print(f"    {'tenths of board reachable':>28} {'games':>7}")
        for k in range(10):
            if hist.get(k):
                print(f"    {k:>10} to {k+1:<15} {hist[k]:>7}")
        tiny = sum(1 for v in tr if v < 0.10)
        print(f"    died with under 10 percent of the board reachable: "
              f"{tiny} of {len(tr)} ({pct(tiny, len(tr)):.0f} percent)")
        small = sum(1 for v in tr if v < 0.25)
        print(f"    died with under 25 percent reachable              : "
              f"{small} of {len(tr)} ({pct(small, len(tr)):.0f} percent)")

    la = r["legal_at_death"]
    if la:
        print()
        print("  LEGAL MOVES AT DEATH")
        c = Counter(la)
        for k in sorted(c):
            print(f"    {k} legal move(s): {c[k]:>3} games ({pct(c[k], len(la)):.0f} percent)")
        forced = sum(1 for v in la if v <= 1)
        print(f"    forced, one move or none: {forced} of {len(la)} "
              f"({pct(forced, len(la)):.0f} percent)")

    tail = r["tail_reachable_at_death"]
    if tail:
        lost = sum(1 for v in tail if not v)
        print()
        print("  TAIL REACHABILITY")
        print(f"    tail unreachable at death: {lost} of {len(tail)} "
              f"({pct(lost, len(tail)):.0f} percent)")
        if r["steps_total"]:
            print(f"    tail unreachable on any step: {r['tail_lost_total']} of "
                  f"{r['steps_total']} steps "
                  f"({pct(r['tail_lost_total'], r['steps_total']):.1f} percent)")

    if r["steps_total"]:
        print()
        print("  WAS THE TRAP AVOIDABLE?")
        print(f"    steps where no move left room for the body: "
              f"{r['trapped_next_step']} of {r['steps_total']} "
              f"({pct(r['trapped_next_step'], r['steps_total']):.1f} percent)")
        print(f"    steps where a safe move existed: "
              f"{r['avoidable_traps']} of {r['steps_total']} "
              f"({pct(r['avoidable_traps'], r['steps_total']):.1f} percent)")

    sa = r["score_at_death"]
    if sa:
        print()
        print("  DEATH BY SCORE BAND")
        print(f"    {'band':>10} {'games':>6} {'self deaths':>12} {'mean trap ratio':>16}")
        for band in sorted(r["by_band"]):
            b = r["by_band"][band]
            mt = np.mean(b["trap"]) if b["trap"] else 0.0
            print(f"    {band:>4} to {band+9:<4} {b['n']:>6} {b['self']:>12} {mt:>16.3f}")

    fpf = r["frames_per_food"]
    if fpf:
        print()
        print("  FRAMES PER FOOD BY SCORE, is it getting harder?")
        print(f"    {'score band':>12} {'mean frames':>13} {'samples':>9}")
        banded = defaultdict(list)
        for score, frames in fpf.items():
            banded[(score // 10) * 10].append(frames)
        for band in sorted(banded):
            vals = [x for sub in banded[band] for x in sub]
            print(f"    {band:>4} to {band+9:<5} {np.mean(vals):>13.1f} {len(vals):>9}")

    hs = r["head_area_series"]
    if hs:
        print()
        print("  HEAD AREA OVER TIME, fraction of the board reachable")
        a = np.asarray(hs)
        for q in (5, 25, 50, 75, 95):
            print(f"    {q:>3}th percentile: {np.percentile(a, q):.3f}")


def main():
    p = argparse.ArgumentParser(description="Trap statistics for Snake agents")
    p.add_argument("--checkpoint")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--all", action="store_true",
                   help="every *_best.pth in model and model/sweep")
    a = p.parse_args()

    paths = []
    if a.checkpoint:
        paths.append(a.checkpoint)
    if a.all:
        paths += sorted(glob.glob(os.path.join(ROOT, "model", "*_best.pth")))
        paths += sorted(glob.glob(os.path.join(ROOT, "model", "sweep", "*_best.pth")))
    if not paths:
        p.print_help()
        return

    for path in paths:
        if not os.path.exists(path):
            print(f"missing: {path}")
            continue
        try:
            report(analyse(path, games=a.games))
        except Exception as e:
            print(f"FAILED on {path}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
