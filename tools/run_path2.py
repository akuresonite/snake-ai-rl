"""Path 2 arms: the raw grid conv net, curriculum, and the reference.

These are the learned-agent experiments built from the research on how to make
an AI master at Snake. The finding that matters is from
https://saheb.github.io/blog/posts/lessons-from-my-rl-experiments-with-snake/ :
a conv net over the raw grid beat a 24 feature DQN, and training on a small
board transferred to a big one once the representation stopped breaking the
size invariance. Our own measurement agreed: the 11 feature state caps every
learned agent near a mean of 28 to 36, no matter how it is trained.

Every arm logs to TensorBoard under s97__<name>, and every arm is scored
greedily on the SAME 40x30 target board under eval/score_mean, so the curves
can be overlaid honestly. Training scores on a 10x10 board mean something very
different from training scores on 40x30, so the eval curve is the comparison.

Arms
----
  basic-refresh    the 11 feature DQN on 40x30, same step budget, the reference
  grid40           conv net over the raw board, trained ON 40x30
  grid12           same net trained on 12x12, scored on 40x30 (zero shot)
  grid10           same net trained on 10x10, scored on 40x30 (zero shot)
  curriculum       train on 12x12, then continue on 40x30, one set of weights

Usage
-----
    python tools/run_path2.py --list
    python tools/run_path2.py --only basic-refresh,grid10
    python tools/run_path2.py --steps 400000 --only grid12

Runs resume automatically: each arm has its own checkpoint pair, so a rerun
continues rather than starting over.
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_PY = r"C:\Users\akure\miniconda3\python.exe"
GPU_PY = r"C:\Users\akure\miniconda3\envs\whisX\python.exe"

STEPS = 800_000
GAMES = 64
EVAL_ARGS = ["--eval-target", "40x30", "--eval-every", "200000",
             "--eval-games", "20"]
LOG_ARGS = ["--log", "both", "--report-every", "10", "--log-every", "2"]


def _arm(name, steps, board, py=BASE_PY, extra=None, games=GAMES):
    """One standard arm: a grid conv net with the shared eval target."""
    return {
        "py": py,
        "args": ["snake_ai_training_vec.py",
                 "--state", "grid",
                 "--board", board,
                 "--games", str(games),
                 "--steps", str(steps),
                 "--checkpoint", f"model/sweep/s97_{name}_latest.pth",
                 "--best-checkpoint", f"model/sweep/s97_{name}_best.pth",
                 "--run-name", f"s97__{name}",
                 ] + (extra or []) + EVAL_ARGS + LOG_ARGS,
    }


ARMS = {
    "basic-refresh": {
        "py": BASE_PY,
        "args": ["snake_ai_training_vec.py",
                 "--state", "basic",
                 "--games", "256",
                 "--steps", str(STEPS),
                 "--checkpoint", "model/sweep/s97_basic_refresh_latest.pth",
                 "--best-checkpoint", "model/sweep/s97_basic_refresh_best.pth",
                 "--run-name", "s97__basic-refresh",
                 ] + EVAL_ARGS + LOG_ARGS,
    },
    "grid40": _arm("grid40", STEPS, "40x30", py=GPU_PY,
                   extra=["--max-memory", "30000"]),
    "grid12": _arm("grid12", STEPS, "12x12"),
    "grid10": _arm("grid10", STEPS, "10x10"),
    # The shaped arms. grid40 learned to circle until the starvation guard
    # killed it, 12 of 12 deaths by starvation, because a 0 reward for doing
    # nothing gives no gradient towards the food. These arms hand that gradient
    # back through the reward instead of through hand coded features.
    "grid40-stepr": _arm("grid40-stepr", STEPS, "40x30", py=GPU_PY,
                         extra=["--max-memory", "30000",
                                "--step-reward", "-0.1"]),
    "grid40-dist": _arm("grid40-dist", STEPS, "40x30", py=GPU_PY,
                        extra=["--max-memory", "30000",
                               "--distance-reward", "1.0"]),
    "grid40-both": _arm("grid40-both", STEPS, "40x30", py=GPU_PY,
                        extra=["--max-memory", "30000",
                               "--step-reward", "-0.05",
                               "--distance-reward", "0.5"]),
    "curriculum12to40": {
        "py": GPU_PY,
        "phases": [
            ["snake_ai_training_vec.py",
             "--state", "grid", "--board", "12x12",
             "--games", str(GAMES), "--steps", str(STEPS // 2),
             "--checkpoint", "model/sweep/s97_curriculum12to40_latest.pth",
             "--best-checkpoint", "model/sweep/s97_curriculum12to40_best.pth",
             "--run-name", "s97__curriculum12to40",
             ] + EVAL_ARGS + LOG_ARGS,
            ["snake_ai_training_vec.py",
             "--state", "grid", "--board", "40x30",
             "--games", str(GAMES), "--steps", str(STEPS // 2),
             "--max-memory", "30000",
             "--checkpoint", "model/sweep/s97_curriculum12to40_latest.pth",
             "--best-checkpoint", "model/sweep/s97_curriculum12to40_best.pth",
             "--run-name", "s97__curriculum12to40",
             ] + EVAL_ARGS + LOG_ARGS,
        ],
    },
}


def run_arm(name, arm, steps_override):
    env = dict(os.environ)
    env["SDL_VIDEODRIVER"] = "dummy"
    env["SDL_AUDIODRIVER"] = "dummy"
    env["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    py = arm["py"]
    if py == BASE_PY:
        # Several CPU arms can run at once, so each one keeps a small thread
        # pool instead of grabbing all 16 cores for a 17k parameter model.
        env["SNAKE_THREADS"] = "4"

    phases = arm.get("phases") or [arm["args"]]
    for idx, args in enumerate(phases):
        if steps_override and "--steps" in args:
            i = args.index("--steps")
            args = list(args)
            args[i + 1] = str(steps_override // len(phases))
        label = f"{name} phase {idx + 1}" if len(phases) > 1 else name
        print("=" * 78)
        print(f"RUN {label}")
        print("=" * 78, flush=True)
        code = subprocess.run([py] + args, cwd=ROOT, env=env).returncode
        print(f"=> {label} exited with code {code}", flush=True)
        if code != 0:
            return code
    return 0


def main():
    p = argparse.ArgumentParser(description="Run the Path 2 learned agent arms")
    p.add_argument("--list", action="store_true")
    p.add_argument("--only", default=None,
                   help="comma separated arm names to run, e.g. grid10,grid12")
    p.add_argument("--steps", type=int, default=None,
                   help="override the per arm step budget")
    a = p.parse_args()

    if a.list:
        for name, arm in ARMS.items():
            phases = arm.get("phases")
            note = (f"{len(phases)} phases"
                    if phases else "1 phase")
            print(f"  {name:20s} {note:10s} python={os.path.basename(os.path.dirname(arm['py']))}")
        return

    names = [x.strip() for x in (a.only or "").split(",") if x.strip()]
    if not names:
        names = list(ARMS)
    for name in names:
        if name not in ARMS:
            print(f"unknown arm {name}, choices: {sorted(ARMS)}")
            sys.exit(2)
        code = run_arm(name, ARMS[name], a.steps)
        if code != 0:
            print(f"=> stopping after {name} failed ({code})")
            sys.exit(code)
    print("all arms done. TensorBoard: http://localhost:6006")


if __name__ == "__main__":
    main()
