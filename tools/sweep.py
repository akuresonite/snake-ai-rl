"""Run the tier 1 combination sweep, with early stopping and smart resume.

Configurations
--------------
target_net, double_dqn and n_step, crossed. LayerNorm is included as a variant
because removing it was tested and did not help, but every combination is here
so the evidence is in one place.

Early stopping
--------------
A run stops when the best score has not improved for --patience training calls.
That is the point: a configuration that has plateaued should not burn the rest
of the night.

Resume
------
A run is only resumed if it still has room to improve. "Room" is decided from
its own checkpoint:

  stopped_early = True    the score went flat and we gave up. Do NOT resume by
                          default, that is the whole meaning of the flag.
  stopped_early = False   it ran out of step budget while still improving, or
                          it was interrupted. Resume it.

--resume decides which to continue:

  improvable  only runs that did not stop early (default)
  all         every run with a checkpoint
  none        always start fresh

So a second pass over a stopped sweep spends its time only where there is still
something to gain.

Usage
-----
    python -m tools.sweep --steps 1500000 --seeds 0,1
    python -m tools.sweep --steps 1500000 --resume improvable
    python -m tools.sweep --list
"""

import argparse
import itertools
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

CKPT_DIR = os.path.join(ROOT, "model", "sweep")

# group number, label, target_net, double_dqn, n_step
CONFIGS = [
    ("s01", "baseline",        False, False, 1),
    ("s02", "target",          True,  False, 1),
    ("s03", "target-double",   True,  True,  1),
    ("s04", "target-n3",       True,  False, 3),
    ("s05", "target-double-n3", True, True,  3),
    ("s06", "target-n5",       True,  False, 5),
    ("s07", "target-double-n5", True, True,  5),
    ("s08", "target-n2",       True,  False, 2),
    ("s09", "target-double-n2", True, True,  2),
]


def ckpt_paths(group, seed):
    base = os.path.join(CKPT_DIR, f"{group}__s{seed}")
    return base + "_latest.pth", base + "_best.pth"


def read_state(latest, best):
    """Read what the checkpoint says about whether to resume."""
    import torch
    for p in (latest, best):
        if os.path.exists(p):
            try:
                d = torch.load(p, map_location="cpu", weights_only=False)
                return {
                    "exists": True,
                    "record": int(d.get("record", 0)),
                    "steps": int(d.get("steps", 0)),
                    "stopped_early": bool(d.get("stopped_early", False)),
                    "path": p,
                }
            except Exception as e:
                return {"exists": True, "error": f"{type(e).__name__}: {e}",
                        "path": p}
    return {"exists": False}


def should_run(state, mode):
    if mode == "none":
        return True, "resume=none, always fresh"
    if not state.get("exists"):
        return True, "no checkpoint yet"
    if state.get("error"):
        return True, f"checkpoint unreadable ({state['error']})"
    if mode == "all":
        return True, "resume=all"
    # improvable
    if state.get("stopped_early"):
        return False, (f"stopped early at record {state['record']}, "
                       f"no room assumed")
    return True, (f"last run did not stop early, record {state['record']}, "
                  f"resuming")


def main():
    p = argparse.ArgumentParser(description="Tier 1 combination sweep")
    p.add_argument("--steps", type=int, default=1_500_000,
                   help="step budget per run (early stopping usually cuts this)")
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--patience", type=int, default=400,
                   help="training calls without a new record before stopping")
    p.add_argument("--min-train-calls", type=int, default=400)
    p.add_argument("--resume", default="improvable",
                   choices=["improvable", "all", "none"])
    p.add_argument("--state", dest="obs_mode", default="basic",
                   choices=["basic", "rich"])
    p.add_argument("--device", default="auto")
    p.add_argument("--log", dest="log_backend", default="both",
                   choices=["both", "tensorboard", "mlflow", "none"])
    p.add_argument("--only", default=None,
                   help="comma list of group numbers, e.g. s01,s02")
    p.add_argument("--list", action="store_true",
                   help="show the plan and what would resume, run nothing")
    a = p.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip() != ""]
    only = set(x.strip() for x in a.only.split(",")) if a.only else None

    import snake_ai_training_vec as vec

    os.makedirs(CKPT_DIR, exist_ok=True)

    plan = []
    for (group, label, tn, dd, ns) in CONFIGS:
        if only and group not in only:
            continue
        for seed in seeds:
            latest, best = ckpt_paths(group, seed)
            state = read_state(latest, best)
            do, why = should_run(state, a.resume)
            plan.append((group, label, tn, dd, ns, seed, latest, best, state, do, why))

    print("=" * 96)
    print(f"TIER 1 SWEEP | {len(plan)} run(s) | {a.steps:,} steps each max | "
          f"resume={a.resume} | patience={a.patience}")
    print("=" * 96)
    print(f"  {'group':6s} {'variant':18s} {'seed':>4} {'best':>5} {'early?':>7} "
          f"{'run?':>5}  why")
    for (g, label, tn, dd, ns, seed, _l, _b, st, do, why) in plan:
        rec = st.get("record", "-") if st.get("exists") else "-"
        early = ("yes" if st.get("stopped_early") else "no") if st.get("exists") else "-"
        print(f"  {g:6s} {label:18s} {seed:>4} {str(rec):>5} {early:>7} "
              f"{'YES' if do else 'no':>5}  {why}")

    todo = [x for x in plan if x[9]]
    if a.list:
        print()
        print(f"{len(todo)} run(s) would start or resume")
        return
    if not todo:
        print()
        print("nothing to do, every run in this plan stopped early")
        return

    print()
    print(f"starting {len(todo)} run(s)")
    results = []
    for i, (g, label, tn, dd, ns, seed, latest, best, st, do, why) in enumerate(todo):
        run_name = vec.snake_run_name(g, obs_mode=a.obs_mode,
                                      target_net=tn, double_dqn=dd,
                                      n_step=ns, seed=seed)
        fresh = not st.get("exists")
        print()
        print("-" * 96)
        print(f"[{i+1}/{len(todo)}] {run_name}  ({'fresh' if fresh else 'resume'})")
        print("-" * 96)
        for f in ((latest,) if fresh else ()):
            if os.path.exists(f):
                os.remove(f)
        t0 = time.perf_counter()
        try:
            vec.train(
                a.games, a.steps, 8, a.device, latest, best,
                save_every=120.0, quiet=False, log_backend=a.log_backend,
                run_name=run_name, log_dir=os.path.join(ROOT, "runs"),
                report_every=20.0, log_every=1.0, obs_mode=a.obs_mode,
                use_norm=True, target_net=tn, double_dqn=dd, n_step=ns,
                patience=a.patience, min_train_calls=a.min_train_calls,
                log_group=g,
            )
        except KeyboardInterrupt:
            print("=> interrupted, stopping the sweep and saving state")
            break
        except Exception as e:
            print(f"=> RUN FAILED: {type(e).__name__}: {e}")
            results.append((run_name, "FAILED", 0, time.perf_counter() - t0))
            continue
        after = read_state(latest, best)
        results.append((run_name, after.get("record", "?"),
                        after.get("stopped_early", "?"),
                        time.perf_counter() - t0))
        print(f"  => record {after.get('record')}, "
              f"stopped_early={after.get('stopped_early')}")

    print()
    print("=" * 96)
    print("SWEEP SUMMARY")
    print("=" * 96)
    print(f"  {'run':40s} {'record':>7} {'early':>7} {'minutes':>8}")
    for name, rec, early, secs in results:
        print(f"  {name:40s} {str(rec):>7} {str(early):>7} {secs/60:>8.1f}")
    if results:
        best = max((r for r in results if isinstance(r[1], int)),
                   key=lambda r: r[1], default=None)
        if best:
            print()
            print(f"  best record overall: {best[0]} with {best[1]}")


if __name__ == "__main__":
    main()
