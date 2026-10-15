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
always wide. A CUDA capable environment is needed:

```
python snake_ai_training_vec.py --games 256 --device cuda
```

Your conda `whisX`, `joycaption` and `pinclass` environments have CUDA torch
that supports this card. None of them have `pygame`, which this script does not
need at all, since it never opens a window.

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
snake_ai_training.py         original single window version
snake_ai_training2.py        dashboard UI, baseline shallow network
snake_ai_training_Deep.py    reference implementation, one game at a time
snake_ai_training_vec.py     many games at once, one network, no window
model/                       checkpoint directory, contents gitignored
requirements.txt             runtime dependencies
.gitattributes               pins LF line endings across platforms
LICENSE                      MIT
```

The owner also keeps a local `AGENTS.md` with contributor rules and the current
configuration. It is deliberately gitignored and not part of this repository.

## License

MIT, see [LICENSE](LICENSE).
