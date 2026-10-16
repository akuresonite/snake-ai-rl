# Snake AI with Deep Q-Learning

A Snake game and a Deep Q-Network agent that learns to play it, written from
scratch on top of PyTorch and Pygame. There is no Gym and no RL framework here,
the environment, the replay buffer, the network and the training loop are all
hand rolled so every moving part stays readable.

## Which script is the good one

The files were written in order and each one supersedes the last. There is one
addition that runs alongside rather than replacing anything, marked below.

| Script | Written | Network | Optimizer | UI |
|--------|---------|---------|-----------|-----|
| `snake_ai_training.py` | Dec 3, 2025 | 11 to 256 to 3, single hidden layer | Adam, lr 0.001 | Single window, matplotlib live plot, fixed 40 fps |
| `snake_ai_training2.py` | Dec 5, 2025 | 11 to 256 to 3, single hidden layer | Adam, lr 0.001 | Three panel dashboard, embedded Plotly chart, speed slider, scrollable history |
| `snake_ai_training_Deep.py` | Dec 6, 2025 | 11 to 128 to 128 to 3, LayerNorm, no dropout | AdamW, lr 0.0005 | Same three panel dashboard as above |
| `snake_ai_training_vec.py` | later addition | same network as the reference | same trainer as the reference | none, headless, many games at once |

**`snake_ai_training_Deep.py` is the superior script and the one to run.** It is a
strict superset of `snake_ai_training2.py`: everything that file does, it does
too, plus a better optimizer, a corrected training step and its own checkpoint
file so the two models can be trained without clobbering each other.

`snake_ai_training2.py` is worth keeping as the baseline. It is the honest
comparison point for any claim about the reference network, because its
environment, rewards, replay buffer and UI are byte for byte the same.

`snake_ai_training.py` is the earliest version and is superseded on every axis.

### How they actually differ

The environment is identical across all three files. So is the 11 value state
vector, the `+10` food reward, the `-10` death reward, the starvation guard at
`100 * len(snake)` frames, the `100000` entry replay buffer and the `1000`
sample batch size.

The divergence is the function approximator, its optimizer and the shape of the
training step:

- **Reference net.** Two hidden layers of 128 units, each followed by
  `LayerNorm` and `ReLU`. No dropout.
- **Baseline net.** One hidden layer of 256 units with `ReLU`.
- **Optimizer.** `AdamW` at `lr=0.0005` with `weight_decay=1e-4` in the
  reference, plain `Adam` at `lr=0.001` in the baseline and the original.
- **Loss.** `SmoothL1Loss` (Huber) in the reference, `MSELoss` in the other two.

### How the reference script was fixed

The first version of the reference script had real defects. They are corrected
now, and the fixes are worth knowing about because they change results:

1. **Dropout was corrupting the target.** The bootstrap target ran with the
   model still in training mode, so dropout noise leaked into the regression
   target. The target now runs under `torch.no_grad()` with the model in
   `eval()` mode. Dropout is also removed entirely, since with a clean target
   its only remaining effect was noise.
2. **One forward pass per batch element.** The target looped over the batch in
   Python, up to 1000 model calls per game over. It is now one vectorized pass
   over the whole batch.
3. **`torch.argmax(action[idx])`** was an indirect way to write the chosen
   action index. It is now computed once for the whole batch.
4. **`torch.load` on modern PyTorch.** The checkpoint carries optimizer state,
   which the new `weights_only=True` default rejects. The loader tries the safe
   path first and falls back.
5. **Exploration collapsed to zero.** The old `80 - n_games` schedule went
   negative and pinned to pure greed. It is now a multiplicative decay to a
   `0.01` floor over 200 games.
6. **The network was oversized.** Three 256 unit layers on an 11 value binary
   input is far more capacity than the problem needs. The reference net is now
   128 by 128, which is 18,947 parameters instead of 136,963.
7. **Recursive food placement.** `_place_food` called itself when it landed on
   the snake, which risks a stack overflow on a nearly full board and loops
   forever on a full one. It is now iterative, and a full board ends the episode.
8. **Float head coordinates.** `Point(self.game_w / 2, ...)` produced floats,
   which makes `Point` equality and hashing fragile. Coordinates are integers.
9. **A missing `kaleido` killed the run** at the first game over, because
   `plotly.io.to_image` raises. Plotly is now optional, a render failure warns
   once and training continues without the chart panel.
10. **Chart rendering blocked training** on every game over. It is throttled to
    every 5 games, and a final PNG is written on Ctrl+C.
11. **Font surfaces were rebuilt every frame.** The score label is now cached
    against the score value.
12. **`is_collision` and the state builder hardcoded `20`.** Both use
    `BLOCK_SIZE` now.

### Known limits that remain

Not defects, just things a reader should know:

- **No target network.** Standard DQN keeps a slowly synced copy of the network
  to compute targets. This code bootstraps from the online network, which is
  less stable. That is the single biggest remaining improvement.
- **The replay buffer is not saved.** A restart loses all memories.
- **No reward shaping.** The reward is still `+10` for food and `-10` for death.
- **Learning rate and size are choices, not proofs.** The reference net has not
  been benchmarked against the baseline in a controlled run.

## Running many games at once

`snake_ai_training_vec.py` trains the same network on many snakes in parallel,
in one process, with one replay buffer.

```
python snake_ai_training_vec.py --games 64 --steps 500000
```

Measured on this machine, 64 parallel games, CPU:

```
finished 500,032 steps in 70.1s
  throughput     : 7,129 steps/s
  games finished : 1936
  best score     : 62
  avg return     : 242.45 (last 200 games)
```

Then the saved model, played greedily with no exploration in 30 fresh games:

```
mean 33.17, best 68, worst 5, 29 of 30 games scored 10 or more
```

For comparison, the single game script trains at about 923 steps per second,
so this is roughly 8x the raw step rate. The bigger effect is that the whole
500,000 step run finishes in about 70 seconds, which is what makes the score
curve actually visible while you watch it.

### Why it is faster, honestly

Two separate effects, and it matters which is which:

1. **Batched work.** One network call handles all 64 games. The vectorized
   environment alone, with no training, runs about 1.4x faster than the single
   game environment alone.
2. **Less training per step.** The single game script trains on every single
   step, one sample at a time. This script trains one replay batch of 1000
   every `--train-every` steps. That is where most of the speed comes from, and
   it is a real tradeoff, not free speed. Lower `--train-every` means more
   learning per step and a slower loop. The default of 8 gives about 1.95
   training samples per environment step, against 1.0 for the single game.

Raising `--games` does not help past a point. The environment is Python and
runs on one core, so 128 games measured barely faster than 64.

### Running it on a GPU

The batched design is what finally makes a GPU worthwhile, because the batch is
always wide. Measured, 200,000 steps, 256 games, same code:

| Device | Throughput |
|--------|------------|
| CPU, base env | 7,387 steps/s |
| GPU, RTX 5050, whisX env | 9,250 steps/s |

That is about 25 percent faster on the GPU. Not a huge margin, and the reason is
worth knowing: the game rules are plain Python and run on one CPU core, so the
GPU spends most of its time waiting. Raising `--games` does not fix that. On the
GPU, 128 games gave 4,434 steps/s and 512 games gave 9,527, so it flattens out
around 256.

To use the GPU you need a Python environment with CUDA torch. On this machine:

```
C:\Users\akure\miniconda3\envs\whisX\python.exe snake_ai_training_vec.py --games 256 --device cuda
```

`whisX`, `joycaption` and `pinclass` all have CUDA torch that supports this
card. None of them have `pygame`, and they do not need it, because this script
imports its shared code from `snake_core.py`, which is deliberately pygame free.
Do not use `audiosr`: its torch reports CUDA as available but has no kernel for
this GPU generation, so it fails at the first real calculation.

### Checkpoints, and why there are two files

The vectorized trainer keeps two files, and they mean different things:

| File | Written when | Use it for |
|------|--------------|------------|
| `model/vec_checkpoint.pth` | every 60 seconds, and on Ctrl+C, and at the end | resuming |
| `model/vec_best.pth` | only when a game sets a new record score | showing off the best model |

They are separate on purpose. A single file cannot be both, and here is the
problem with trying. The record score can be set by a **lucky random game**
early in training. Because a single file only gets written when a record falls,
that lucky value then blocks any save until a genuinely good model beats it. So
on Ctrl+C you would lose everything back to that lucky game.

Two consequences worth knowing:

- **Ctrl+C is safe.** The handler saves the latest state before exiting. Before
  this change, stopping a long GPU run threw away everything since the last
  record.
- **The best file is not a snapshot of the record game.** The save happens at
  the end of the game, after the model has already been trained on that game.
  So it holds the weights just after the record run, not the weights that
  produced it. That is usually better, since the model keeps improving.

Saves are atomic. The file is written to a temporary path and then renamed, so
an interrupted write leaves either the old file or the new one, never a broken
one that fails to load.

Resuming checks the latest file first and falls back to the best file, so either
one can start a run.

### Watching the training curves

The vectorized trainer logs to TensorBoard and MLflow. No install is needed for
MLflow, because the run format on disk is plain JSON and TSV and the trainer
writes it directly. TensorBoard does need one package.

```
python snake_ai_training_vec.py --games 256 --steps 2000000 --run-name first-real-run
```

Then, in another terminal:

```
pip install tensorboard
tensorboard --logdir runs
```

For MLflow, if you have the package:

```
mlflow ui --backend-store-uri runs/mlruns
```

If you do not have the mlflow package, the files are still written and the UI
will read them the moment you install it. Nothing is lost.

**TensorBoard showing an empty page.** The usual cause is that no run has
happened yet, so `runs/` is empty or missing. TensorBoard does not warn about
this, it just shows nothing. Check first:

```
dir runs
```

If it is empty, run training once and then refresh the browser. Give TensorBoard
about 30 seconds to load. Always start it from the project root, with the log
directory as `runs`:

```
tensorboard --logdir runs
```

To confirm TensorBoard can actually see your metrics, ask its own API:

```
curl "http://localhost:6006/data/plugin/scalars/tags"
```

You should get a list of metric names back. If that list is empty, the problem is
that there is no event file, not that the browser is stale.

The trainer prints a note at startup when the run folder was empty before, so
this is visible in the terminal now rather than only in the browser.

Useful flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--steps` | `500000` | how many steps to run **this time**, not a lifetime total |
| `--log` | `both` | `both`, `tensorboard`, `mlflow` or `none` |
| `--run-name` | `run-<timestamp>` | name shown in both UIs |
| `--log-dir` | `./runs` | where the output goes |
| `--log-every` | `1` | seconds between points on the curves |
| `--state` | `basic` | `basic` 11 features, or `rich` 21 features |
| `--report-every` | `5` | seconds between console reports |

`--steps` is a budget for the current run. Resuming from a 2 million step
checkpoint with `--steps 1500000` runs 1.5 million **more** steps, it does not
conclude that the target was already reached. The curves use the cumulative step
count so they line up across runs, and `progress/run_steps` gives the per run
count. A run that resumed correctly prints both numbers:

```
=> resuming after 2,000,128 previous steps, running 1,500,000 more
finished after 1,500,160 steps in this run, 3,500,288 total, in 77.9s
```

The two intervals are separate on purpose. The curves want frequent points, the
terminal only wants to be readable.

`--quiet` silences the terminal only. The curves still get written.

### What gets logged

| Curve | Meaning |
|-------|---------|
| `episode/score` | one point per finished game, the noisy but honest one |
| `episode/best_score` | record score when it improves |
| `episode/score_mean_200` | rolling mean over the last 200 games |
| `episode/score_max_200` | best of the last 200 games |
| `episode/return_mean_200` | rolling mean of the shaped reward |
| `episode/length_mean_200` | average frames survived |
| `explore/epsilon` | exploration rate, should fall to the floor |
| `train/loss_mean` | TD loss, averaged over the report window |
| `train/grad_norm_mean` | gradient norm, see the plateau note below |
| `train/q_mean_mean`, `train/q_max_mean`, `train/target_mean_mean` | value estimates |
| `throughput/steps_per_s` | speed, so you can see a regression when you change something |
| `replay/size`, `replay/train_calls` | buffer filling up |
| `progress/steps` | cumulative across every run, the axis for the curves |
| `progress/run_steps` | steps in the current run only |
| `progress/episodes`, `progress/wall_clock_s` | episode axis and timing |

Diagnostics are averaged into one point per log window. Writing them every
training step produced about 49,000 rows per metric in a 150,000 step run, which
made the MLflow store slow to open for no extra information.

### Two state vectors, and which one to use

`--state basic` (the default) gives the agent 11 features: danger in three
directions, the current direction, and which way the food is.

`--state rich` gives it 21: the same 11, plus scaled distance to each wall,
which way the tail is, the snake length, and how far away the food is.

Measured, 1.2 million steps, 256 games, 4 seeds each:

| state | features | mean best score | scores |
|-------|----------|-----------------|--------|
| basic | 11 | **63.0** | 56, 63, 66, 67 |
| rich | 21 | 59.8 | 55, 59, 67, 58 |

The richer state is about 3 points worse, and it costs roughly 25 percent of the
throughput. So the default stays `basic`. The extra information is real, it just
does not help here.

One of the extra features is provably dead weight. The snake length feature sits
at a constant 1 for the early game, because a random snake dies within about 5
moves so its length never gets far past 3. Measured over 256,000 observations,
raw length stayed between 3 and 5.

The honest conclusion: this agent is not short of information, so handing it more
is not the way past the plateau. The limit is in how it learns, not what it sees.

### Advanced methods, and early stopping

Four switches, all off by default so existing runs and checkpoints are unchanged.

| Flag | What it does |
|------|--------------|
| `--target-net` | A lagged copy of the network computes the bootstrap target, Polyak averaged towards the online net by `--tau`. The standard DQN fix. |
| `--double-dqn` | The online net picks the next action, the target net scores it. Cuts the overestimation a plain `max()` causes. Needs `--target-net`. |
| `--n-step N` | Sums the reward over N steps when filling the replay buffer, and discounts by `gamma ** N`. Propagates the sparse food reward faster. |
| `--patience N` | Stop when the best score has not improved for N training calls. |

**Early stopping is the important one.** A configuration that has flattened out
should not spend the rest of the night running. Measured here, a 1.5 million step
budget stopped after about 870,000 steps once the score went flat, roughly 45
seconds instead of minutes per configuration.

```
python snake_ai_training_vec.py --target-net --double-dqn --n-step 3 --patience 400
```

The whole sweep is scripted:

```
python -m tools.sweep --steps 1500000 --seeds 0,1 --patience 400
python -m tools.sweep --list
```

### Resuming only where there is room

A resume is only worth doing where a run still had room to improve. Each
checkpoint records whether its own run gave up early:

| `stopped_early` | Meaning | Resumed by default? |
|---|---|---|
| `True` | the score went flat and the run stopped itself | no, that is what the flag means |
| `False` | it used its whole step budget while still improving, or was interrupted | yes |

`--resume` chooses `improvable` (default), `all` or `none`:

```
python -m tools.sweep --resume improvable
```

So a second pass over a stopped sweep spends its time only where something can
still be gained.

### Reading the TensorBoard sidebar

Run folders are named so the list sorts sensibly and reads clearly:

```
<group>__<variant>__s<seed>
```

| Example | Meaning |
|---|---|
| `s01__baseline__s0` | plain DQN, seed 0 |
| `s03__target-double__s1` | target network plus Double DQN, seed 1 |
| `s05__target-n3__s2` | target network plus 3 step returns, seed 2 |
| `s90__approx-forest__s0` | random forest as the Q function, seed 0 |

Groups sort in sweep order and all seeds of one variant sit together. Finished
runs move to `runs/archive` so the top level stays readable.

```
python tools/rename_runs.py --plan --archive-old
python tools/rename_runs.py --apply --archive-old
```

That tool refuses to touch a run that is still being written, because moving a
folder out from under a live training process breaks its open file handle.
`runs/README.txt` explains the scheme in place.

### Other Q functions, not just a neural net

`snake_approx_compare.py` runs the same environment, replay buffer, epsilon
schedule, target rule and rewards, and swaps only the thing that turns a state
into Q values. Every approximator logs the same metric names, so the curves
overlay in TensorBoard.

```
python snake_approx_compare.py --models dqn,logreg,forest,tree,svm,mlp,lightgbm \
    --steps 400000 --games 256
```

| Model | Notes |
|-------|-------|
| `dqn` | the torch network, trained every step |
| `logreg` | LogisticRegression, one per action |
| `forest` | RandomForestRegressor |
| `tree` | DecisionTreeRegressor |
| `svm` | SVR with an RBF kernel and scaling |
| `mlp` | sklearn MLPRegressor, one hidden layer of 64 |
| `mlp2` | sklearn MLPRegressor, 128 by 128 |
| `lightgbm` | LGBMRegressor |

At 400,000 steps and one seed the forest beat the DQN, 31 against 21. One seed
is not a conclusion, but it says the neural net is not obviously the right
approximator for an 11 bit state.

### Laya, and what can honestly be asked of it

`laya_snake.py` runs `convaiinnovations/laya`, requested for this project. It is
worth being precise about that model, because it changes the question.

Laya is a **non autoregressive text decision model**: a 421M parameter
ModernBERT-large backbone, trained with reinforcement learning to return
calibrated probabilities for **typed questions about text**. It has no vision
input and no action output. It cannot look at a board and it cannot emit a
direction.

So the board is rendered as text and Laya is asked, in plain language, which of
the three legal moves is safe. That is a fair demonstration on the game, and the
honest expectation is that it plays badly, because Snake needs planning several
moves ahead while a text classifier scores one option list in one forward pass.
A bad score there is a real result about the model, not a bug.

```
C:\Users\akure\miniconda3\envs\whisX\python.exe laya_snake.py --decision --episodes 5
C:\Users\akure\miniconda3\envs\whisX\python.exe laya_snake.py --policy
```

`whisX` is the environment that works: Python 3.10, CUDA torch, transformers
4.57, plus `laya` 0.4.8 and `tensorboard`. Measured: 638 ms per decision on the
RTX 5050 for a 171 token board.

The second experiment plays to what Laya is actually good at. Ask it to choose a
**strategy**, then run that strategy. It chooses between flood fill, greedy and
random, and the chosen one is executed and measured. That separates "can this
model play Snake" from "can this model usefully decide something about Snake".

### The raw grid conv net, small boards, and what Path 2 showed

`tools/run_path2.py` runs the learned-agent arms built from the external
research on making an AI master Snake. The idea under test: a conv net over the
raw grid should beat the hand coded 11 feature DQN, and a policy trained on a
small board should transfer to a big one through global average pooling.

```
python tools/run_path2.py --list
python tools/run_path2.py --only grid40-stepr,grid40-dist,grid40-both
```

Every arm runs 800,000 environment steps and every arm is scored greedily on
the same 40x30 target board, which is what makes the curves comparable.

| arm | trained on | train best | eval 40x30 mean | eval best |
|-----|-----------|-----------|-----------------|-----------|
| basic-refresh | 40x30, 11 features | 54 | **27.65** | **49** |
| grid40 | 40x30, raw grid CNN | 3 | 0.00 | 0 |
| grid10 | 10x10, raw grid CNN | 16 | 0.00 | 0 |
| grid12 | 12x12, raw grid CNN | 11 | 0.05 | 1 |
| curriculum12to40 | 12x12 then 40x30 | 6 | 0.05 | 1 |
| grid40-stepr | 40x30, step penalty | 3 | 0.10 | 1 |
| grid40-dist | 40x30, distance shaping | 26 | 2.90 | 9 |
| grid40-both | 40x30, both | 24 | 3.50 | 9 |

The full story, in one paragraph. The raw grid conv net learns to circle
without eating on the big board: measured deaths are 12 of 12 by the starvation
guard, with the food about 14 cells away on average, because a zero reward for
doing nothing leaves no gradient towards the food. Training on 10x10 and 12x12
works there, but the zero shot transfer to 40x30 scores 0, because after two
max pools the receptive field covers about 11 by 11 cells and the global
pooling carries no distance, so far food is invisible. A step penalty alone
does not help. Distance shaping is the only fix that works, and it confirms the
missing ingredient: those arms are the first CNN runs to eat on the big board,
at eval mean 2.90 and 3.50. The 11 feature DQN still beats them eight to ten
times over, because the four food direction bits hand over the food gradient
for free. The external finding that a conv net beats hand features does not
transfer to a 40x30 board with sparse plus/minus 10 rewards, and the reason is
now measured, not guessed.

The planning result remains the strongest number in the repo: `policy_bfs_safe`
in `tools/search_baseline.py` scores mean 132 with no learning at all.

## Requirements

```
pip install -r requirements.txt
```

Which covers:

```
torch
pygame
numpy
plotly
kaleido
```

Optional, for Intel GPU acceleration:

```
intel_extension_for_pytorch
```

The code targets Python 3.13 and up. It picks its compute device automatically:
CUDA first, then Intel XPU if IPEX is present, then CPU.

On this machine the working interpreter is the conda one, because the Python on
`PATH` has no torch:

```bash
C:\Users\akure\miniconda3\python.exe snake_ai_training_Deep.py
```

That interpreter reports torch 2.8.0, pygame 2.6.1 and numpy 2.2.6.

## Running it

```bash
python snake_ai_training_Deep.py
```

Training starts immediately and resumes from `model/deep_checkpoint.pth` if that
file exists and matches the current network shape. Press Ctrl+C to stop, which
saves the current state on the way out.

If the checkpoint was saved by a different network shape, the script prints both
shapes and starts fresh. It never half loads, and it never overwrites the old
file unless training reaches a new record.

The window is split into three panels: training history on the left, the game in
the middle, and a live Plotly chart of score and mean score on the right. Drag
the slider under the chart to change the simulation speed, and scroll over the
left panel to browse older games.

Checkpoints are written whenever the record score improves, and every ten games
regardless. Model weights are gitignored, so you will not find them in this repo.

## Layout

```
snake_core.py                shared pygame free core, imported by both below
snake_logging.py             TensorBoard and MLflow metrics
snake_ai_training.py         original single window version
snake_ai_training2.py        dashboard UI, baseline shallow network
snake_ai_training_Deep.py    reference implementation, one game at a time
snake_ai_training_vec.py     many games at once, one network, no window
model/                       checkpoint directory, contents gitignored
runs/                        TensorBoard and MLflow output, gitignored
requirements.txt             runtime dependencies
.gitattributes               pins LF line endings across platforms
LICENSE                      MIT
```

`snake_core.py` holds the network, the trainer, every hyperparameter and the
checkpoint helpers. It imports no pygame, on purpose. The single game script
adds the window and the chart, and the vectorized script adds the parallel
environment. A checkpoint written by either script loads in the other, because
the architecture fingerprint is shared.

The owner also keeps a local `AGENTS.md` with contributor rules and the current
configuration. It is deliberately gitignored and not part of this repository.

## License

MIT, see [LICENSE](LICENSE).
