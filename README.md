# Snake AI with Deep Q-Learning

A Snake game and a Deep Q-Network agent that learns to play it, written from
scratch on top of PyTorch and Pygame. There is no Gym and no RL framework here,
the environment, the replay buffer, the network and the training loop are all
hand rolled so every moving part stays readable.

## Which script is the good one

Three iterations live side by side. They were written in order and each one
supersedes the last.

| Script | Written | Lines | Network | Optimizer | UI |
|--------|---------|-------|---------|-----------|-----|
| `snake_ai_training.py` | Dec 3, 2025 | 490 | 11 to 256 to 3, single hidden layer | Adam, lr 0.001 | Single window, matplotlib live plot, fixed 40 fps |
| `snake_ai_training2.py` | Dec 5, 2025 | 602 | 11 to 256 to 3, single hidden layer | Adam, lr 0.001 | Three panel dashboard, embedded Plotly chart, speed slider, scrollable history |
| `snake_ai_training_Deep.py` | Dec 6, 2025 | 628 | 11 to 256 to 256 to 256 to 3, LayerNorm and dropout | AdamW, lr 0.01 | Same three panel dashboard as above |

**`snake_ai_training_Deep.py` is the superior script and the one to run.** It is a
strict superset of `snake_ai_training2.py`: everything that file does, it does
too, plus a deeper regularized network, a better optimizer and its own checkpoint
file so the two models can be trained without clobbering each other.

`snake_ai_training2.py` is worth keeping as the baseline. It is the honest
comparison point for any claim about the deep network, because its environment,
rewards, replay buffer and UI are byte for byte the same.

`snake_ai_training.py` is the earliest version and is superseded on every axis.

### How they actually differ

The environment is identical across all three files. So is the 11 value state
vector, the `+10` food reward, the `-10` death reward, the starvation guard at
`100 * len(snake)` frames, the `100000` entry replay buffer and the `1000`
sample batch size.

The only real divergence is the function approximator and its optimizer:

- **Reference net.** Three hidden layers of 256 units, each followed by
  `LayerNorm` and `ReLU`, with `Dropout(0.1)` between layers. Four `Linear`
  layers total.
- **Baseline net.** One hidden layer of 256 units with `ReLU`. Two `Linear`
  layers total.
- **Optimizer.** `AdamW` at `lr=0.01` in the reference, plain `Adam` at
  `lr=0.001` in the baseline and the original.

### Honest caveats about the reference script

Being the best of the three is not the same as being optimal. These are real
weaknesses, listed so nobody mistakes the lineage for a finished design.

1. **Dropout is active while the target Q-values are computed.** In
   `QTrainer.train_step`, the bootstrap target calls `self.model(next_state[idx])`
   with the model still in training mode, so dropout noise leaks into the
   regression target. Wrapping that call in `torch.no_grad()` and a `model.eval()`
   block is the standard fix and would likely improve stability more than the
   extra depth does.
2. **The network is oversized for the problem.** An 11 value binary input on an
   88 cell grid does not need three 256 unit layers with regularization. For
   tabular scale state spaces this much capacity tends to slow learning down
   rather than help it.
3. **`torch.argmax(action[idx])` is redundant.** `action` is already a one hot
   vector, so the argmax is just an indirect way of writing the chosen index.
   `action[idx].item()` says the same thing directly.
4. **Per sample forward passes in a Python loop.** The target computation calls
   the model once per batch element. A single vectorized forward pass over
   `next_state` would be dramatically faster.
5. **The epsilon schedule is tied to game count, not steps.** `80 - n_games`
   compared against `randint(0, 200)` means exploration is fully annealed after
   80 games, and it goes to zero rather than holding at a floor.
6. **Head coordinates are floats.** `Point(self.game_w / 2, ...)` yields float
   coordinates, which works but is fragile. Integer division would be safer.
7. **The dashboard needs `kaleido`.** `plotly.io.to_image` raises on the first
   game over without it, and the call blocks the training loop while it renders.
8. **Loading a checkpoint needs `weights_only=False` on modern PyTorch.** The
   checkpoint dict contains optimizer state, so the default in recent versions
   will refuse it.

## Requirements

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

The code targets Python 3.14 and picks its compute device automatically: CUDA
first, then Intel XPU if IPEX is present, then CPU.

## Running it

```bash
python snake_ai_training_Deep.py
```

Training starts immediately and resumes from `model/deep_checkpoint.pth` if that
file exists. Press Ctrl+C to stop, which saves the current state on the way out.

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
snake_ai_training_Deep.py    reference implementation
model/                       checkpoint directory, contents gitignored
AGENTS.md                    contributor rules and configuration
```

## License

No license file yet. Treat the code as all rights reserved until one is added.
