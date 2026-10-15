# AGENTS.md

Operating rules for any AI agent or human contributor working in this repository.
This file is the single source of truth for conventions, configuration and the
hard rules the owner has laid down. It is append-only in spirit: new rules get
added, nothing gets silently dropped.

Owner: Ashish Kumar Uchadiya (GitHub: [akuresonite](https://github.com/akuresonite))

---

## 1. Hard rules

These are non-negotiable. If a rule here conflicts with a tool default, the rule wins.

| # | Rule | Detail |
|---|------|--------|
| R1 | No em dashes, anywhere | Never emit the U+2014 character. Not in commit messages, not in code comments, not in docs, not in chat replies. Use a comma, a colon, parentheses or a plain hyphen instead. |
| R2 | No AI attribution in commits | Never add `Co-Authored-By:`, `Generated-By:`, `Assisted-By:` or any similar trailer naming an AI agent, model or tool. Commits are authored by the human only. |
| R3 | Humanized commit messages | Write commit subjects the way a person describes the change out loud. Imperative mood, sentence case, no conventional-commit prefixes, no ticket-speak. Body only when it earns its place. |
| R4 | Keep this file current | Whenever the owner states a new rule, config value or preference, append it here in the same session. Do not wait to be asked twice. |
| R5 | Repo is public | This repository is published publicly under `akuresonite`. Never commit secrets, tokens, credentials, personal data or anything that cannot be shown in public. |

### Commit message style guide

Good:

```
Add checkpoint resume so long training runs survive a crash
```

```
Fix food spawning on top of the snake tail
```

```
Swap matplotlib for an embedded plotly panel
```

Bad, and why:

```
feat: add checkpoint resume   <- conventional-commit prefix, rule R3
Add checkpoint resume - ...   <- em dash, rule R1
Add checkpoint resume ...     <- missing co-author trailer, rule R2
```

Keep subject lines under about 72 characters. Commit one logical change at a
time. Never bundle unrelated edits into a single commit.

---

## 2. What this repository is

A from-scratch reinforcement learning playground where a Deep Q-Network learns
to play Snake. No Gym, no Stable Baselines: the environment, the replay buffer,
the network and the training loop are all hand rolled so the moving parts stay
visible.

### Canonical implementation

`snake_ai_training_Deep.py` is the reference script and the one to run. The
other two files are kept as lineage, not as alternatives:

| File | Role | Status |
|------|------|--------|
| `snake_ai_training_Deep.py` | Reference. Deep Q-network with LayerNorm and dropout, AdamW, full dashboard UI. | Active |
| `snake_ai_training2.py` | Baseline. Same dashboard, shallow two-layer net, plain Adam. | Kept for comparison |
| `snake_ai_training.py` | Original. Minimal single-window UI with a matplotlib live plot. | Kept for history |

If you touch training logic, change it in the reference script first, then decide
whether the baseline deserves the same fix.

---

## 3. Environment and configuration

| Setting | Value |
|---------|-------|
| Language | Python 3.14 (developed against 3.14.7) |
| Core libraries | `torch`, `pygame`, `numpy`, `plotly` |
| Plot export backend | `kaleido`, required by `plotly.io.to_image` |
| Default branch | `main` |
| Device selection | `get_device()` prefers CUDA, then Intel XPU via `intel_extension_for_pytorch`, then CPU |

### Reference script hyperparameters

| Constant | Value | Where |
|----------|-------|-------|
| `MAX_MEMORY` | 100000 | replay buffer cap |
| `BATCH_SIZE` | 1000 | experience replay sample size |
| `LR` | 0.01 | AdamW learning rate |
| `gamma` | 0.9 | discount factor |
| `hidden_size` | 256 | three hidden layers in the reference net |
| `dropout_p` | 0.1 | dropout between hidden layers |
| epsilon schedule | `80 - n_games`, compared against `randint(0, 200)` | exploration decay |
| `CHECKPOINT_PATH` | `./model/deep_checkpoint.pth` | reference checkpoint |
| Food reward | `+10` | `play_step` |
| Death reward | `-10` | collision or starvation guard |
| Starvation guard | `frame_iteration > 100 * len(snake)` | `play_step` |

### Running it

```bash
python snake_ai_training_Deep.py
```

Training resumes automatically if `model/deep_checkpoint.pth` exists. Press
Ctrl+C to stop, which saves the current state before exiting.

Note: `pio.to_image` needs `kaleido` installed, otherwise the dashboard plot
panel fails on the first game over.

---

## 4. Working agreement

1. Read this file before making changes.
2. Prefer small, reviewable commits that each do one thing.
3. Run the script at least until the first game over before claiming a change works.
4. Never commit model weights or checkpoints. `.gitignore` already covers `*.pth`
   and `*.pt`.
5. When a rule is added by the owner, append it to section 1 in the same session.

---

## 5. Change log of rules

| Date | Rule added |
|------|-----------|
| 2026-04-24 | R1 no em dashes, R2 no AI attribution in commits, R3 humanized commit messages, R4 keep this file current, R5 repo is public |
