"""Rename and archive TensorBoard run folders so the sidebar stays readable.

TensorBoard shows each run folder under the logdir as a row in the sidebar,
sorted alphabetically. After a night of sweeps that list turns into a wall of
`run-20261016-011435` style names, which tells you nothing.

This tool applies the same naming scheme the trainer uses:

    <group>__<variant>__s<seed>

and can move finished runs out of the way into runs/archive, so the active
sweep is the only thing in the top level list.

Usage
-----
    python tools/rename_runs.py --list
    python tools/rename_runs.py --plan
    python tools/rename_runs.py --apply
    python tools/rename_runs.py --apply --archive-old

Nothing here deletes data. A rename is a folder move, and the MLflow store is
left alone because MLflow identifies runs by uuid, not by name.
"""

import argparse
import json
import os
import shutil
import sys

# Repo root is one level up from this file.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
ARCHIVE = os.path.join(RUNS, "archive")

# Old name -> new name, for runs that already exist. Written out explicitly
# rather than guessed, so nothing is renamed by accident.
LEGACY_MAP = {
    # name                        new name                    what it was
    "first-run":                 "s00__early-basic-1p5M",
    "first-real-run":            "s00__early-basic-3p5M",
    "run-20261016-004854":       "s00__user-rich-long",
    "run-20261016-005559":       "s00__user-rich-mid",
    "cmp-dqn-s0":                "s10__approx-dqn__s0",
    "cmp-logreg-s0":             "s10__approx-logreg__s0",
    "cmp-forest-s0":             "s10__approx-forest__s0",
    "vs-dqn-s0":                 "s10__approx-dqn-400k__s0",
    "vs-logreg-s0":              "s10__approx-logreg-400k__s0",
    "vs-forest-s0":              "s10__approx-forest-400k__s0",
    "ovh-dqn-s0":                "s00__logging-overhead-dqn",
    "smoke":                     "s00__smoke",
    "smoke2":                    "s00__smoke2",
    "final":                     "s00__smoke-final",
    "tbtest":                    "s00__tensorboard-write-test",
    "guardtest":                 "s00__emptydir-guard-test",
    "unit-test":                 "s00__logger-unit-test",
    "bad-values":                "s00__logger-nan-test",
}

# Never touch these. A run that is still being written must not be moved.
PROTECTED_PREFIXES = ("archive", "mlruns")


def list_runs():
    if not os.path.isdir(RUNS):
        print(f"no runs folder at {RUNS}")
        return []
    out = []
    for name in sorted(os.listdir(RUNS)):
        path = os.path.join(RUNS, name)
        if not os.path.isdir(path):
            continue
        has_tb = os.path.isdir(os.path.join(path, "tensorboard"))
        n_events = 0
        size = 0
        if has_tb:
            for f in os.listdir(os.path.join(path, "tensorboard")):
                if f.startswith("events.out.tfevents"):
                    n_events += 1
                    size += os.path.getsize(os.path.join(path, "tensorboard", f))
        out.append({"name": name, "tensorboard": has_tb,
                    "events": n_events, "bytes": size, "path": path})
    return out


def is_protected(name):
    return any(name.startswith(p) for p in PROTECTED_PREFIXES)


def is_active(run, within_seconds=300):
    """True if this run is still being written.

    A run whose newest event file was touched in the last few minutes must not
    be moved. Moving a folder out from under a live training process breaks its
    open file handle. That happened on a dry run of this tool: it wanted to
    archive the owner's active run.
    """
    import time
    newest = 0.0
    for root, _dirs, files in os.walk(run["path"]):
        for f in files:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(root, f)))
            except OSError:
                pass
    return (time.time() - newest) < within_seconds


def plan(archive_old=True):
    """Work out the moves without doing any of them."""
    moves = []
    skipped_active = []
    for r in list_runs():
        name = r["name"]
        if is_protected(name):
            continue
        if is_active(r):
            # Still being written. Renaming or moving it would break the live
            # process, so leave it alone and say so.
            skipped_active.append(name)
            continue
        new = LEGACY_MAP.get(name)
        # Runs that already follow the scheme stay where they are.
        already_named = "__" in name
        if new is None and already_named:
            continue
        if new is None:
            # Unknown name. Leave it in place rather than guessing a new one.
            continue
        dest_dir = ARCHIVE if (archive_old and not already_named) else RUNS
        src = os.path.join(RUNS, name)
        dst = os.path.join(dest_dir, new)
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        if os.path.exists(dst):
            print(f"  SKIP  {name} -> {os.path.relpath(dst, ROOT)} already exists")
            continue
        moves.append((name, new, src, dst))
    return moves, skipped_active


def apply(moves, dry=False):
    ok = 0
    for name, new, src, dst in moves:
        rel = os.path.relpath(dst, ROOT)
        if dry:
            print(f"  would move  {name:26s} -> {rel}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            shutil.move(src, dst)
            print(f"  moved       {name:26s} -> {rel}")
            ok += 1
        except OSError as e:
            print(f"  FAILED      {name}: {type(e).__name__}: {e}")
    return ok


def write_index():
    """Drop a small README into runs/ explaining the scheme."""
    path = os.path.join(RUNS, "README.txt")
    text = (
        "TensorBoard run names\n"
        "=====================\n\n"
        "Scheme: <group>__<variant>__s<seed>\n\n"
        "  group    what is being compared. s01, s02 and so on, so the groups\n"
        "           sort in the order the sweep ran them.\n"
        "  variant  what makes the run different: baseline, target, double,\n"
        "           n3, no-norm, rich, and so on. Dashes, no spaces, because\n"
        "           TensorBoard splits a name on slashes.\n"
        "  seed     the random seed.\n\n"
        "Examples\n"
        "  s01__baseline__s0        plain DQN, seed 0\n"
        "  s03__target-double__s1   target network plus Double DQN, seed 1\n"
        "  s05__target-n3__s2       target network plus 3 step returns, seed 2\n"
        "  s90__approx-forest__s0   random forest Q function, seed 0\n\n"
        "Old runs live in runs/archive so the top level stays readable.\n\n"
        "To view everything, point TensorBoard at runs, which includes archive.\n"
        "To view only the current sweep, point it at runs and ignore archive,\n"
        "or move archive out of the way.\n\n"
        "Regenerate this naming with: python tools/rename_runs.py --apply\n"
    )
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {os.path.relpath(path, ROOT)}")


def main():
    p = argparse.ArgumentParser(description="Rename TensorBoard run folders")
    p.add_argument("--list", action="store_true", help="show what is there")
    p.add_argument("--plan", action="store_true", help="show what would move")
    p.add_argument("--apply", action="store_true", help="do the moves")
    p.add_argument("--archive-old", action="store_true",
                   help="move renamed legacy runs into runs/archive")
    p.add_argument("--index", action="store_true",
                   help="write runs/README.txt explaining the scheme")
    a = p.parse_args()

    if a.list or not (a.plan or a.apply or a.index):
        print(f"runs folder: {RUNS}")
        print(f"{'name':28s} {'tb':>4} {'events':>7} {'bytes':>10}")
        for r in list_runs():
            print(f"  {r['name']:26s} {'yes' if r['tensorboard'] else 'no':>4} "
                  f"{r['events']:>7} {r['bytes']:>10,}")
        if not (a.plan or a.apply or a.index):
            return

    if a.plan or a.apply:
        moves, skipped = plan(archive_old=a.archive_old)
        if skipped:
            print(f"left alone, still being written: {', '.join(skipped)}")
        if not moves:
            print("nothing to move")
        else:
            print(f"{'would move' if a.plan else 'moving'} {len(moves)} folder(s):")
            apply(moves, dry=not a.apply)

    if a.index:
        write_index()


if __name__ == "__main__":
    main()
