"""Shared, pygame free core for the Snake DQN scripts.

Both snake_ai_training_Deep.py and snake_ai_training_vec.py import from here.
Nothing in this file touches pygame, so a CUDA environment without pygame
installed can still train. That is the whole point of the split: the vectorized
trainer once imported the single game script, which imports pygame at line 1,
and died on a machine that had everything except pygame.

Keep it that way. If a pygame import ever appears in this file, the vectorized
trainer stops working on those environments.
"""

import os
from collections import namedtuple
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Try importing Intel Extension for PyTorch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    ipex = None


# ==========================================
# DEVICE CONFIGURATION
# ==========================================

def get_device(label=True):
    """Pick the best available device.

    CUDA only appears here if the installed torch was built with CUDA support.
    A wheel whose version ends in +cpu will always fall through to CPU, no
    matter what hardware is in the machine.
    """
    if torch.cuda.is_available():
        if label:
            print(f"Using CUDA Device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    try:
        if ipex and hasattr(torch, "xpu") and torch.xpu.is_available():
            if label:
                print(f"Using Intel XPU: {torch.xpu.get_device_name(0)}")
            return torch.device("xpu")
    except Exception:
        pass

    if label:
        print("Using CPU")
    return torch.device("cpu")


DEVICE = get_device()


# ==========================================
# THREAD CONFIGURATION
# ==========================================
# Measured on an RTX 5050 laptop with the 11 -> 128 x2 -> 3 network, single
# sample steps:
#
#   1 thread  : 1595 us per step
#   2 threads : 2303 us
#   4 threads : 2603 us
#   8 threads : 2874 us
#   16 threads: 2610 us   (the default)
#
# More threads made the single sample step slower. The model is tiny, so the
# thread pool setup costs more than the arithmetic it saves.
#
# A wide batched step is the opposite case and may benefit from threads, so this
# module does not force a value. Set SNAKE_THREADS to choose one. Each script
# sets its own default.
def set_threads(n):
    """Apply a thread count and return it. Never raises."""
    try:
        torch.set_num_threads(int(n))
        return int(n)
    except Exception:
        return None


SNAKE_THREADS = os.environ.get("SNAKE_THREADS")


# ==========================================
# HYPERPARAMETERS
# ==========================================
# Kept in one place so the values in AGENTS.md and the README stay honest.

# Network shape. The state is 11 binary values on an 88 cell grid, which is
# close to tabular, so a small network learns faster here than a big one.
HIDDEN_SIZE = 128
HIDDEN_LAYERS = 2
DROPOUT_P = 0.0        # 0 disables dropout entirely

# Optimisation
LR = 5e-4
WEIGHT_DECAY = 1e-4
GAMMA = 0.9
MAX_MEMORY = 100_000
BATCH_SIZE = 1000

# Exploration. Multiplicative decay with a floor, rather than a linear ramp
# that reaches zero and then stays there.
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY_GAMES = 200

# Rewards and episode limits
REWARD_FOOD = 10.0
REWARD_DEATH = -10.0
STARVATION_FRAMES_PER_SEGMENT = 100

# Board geometry
BLOCK_SIZE = 20


# ==========================================
# GAME TYPES, no pygame
# ==========================================

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

# Clockwise order, used by every turn calculation.
CLOCKWISE = (Direction.RIGHT.value, Direction.DOWN.value, Direction.LEFT.value, Direction.UP.value)


# ==========================================
# THE BRAIN
# ==========================================

class Linear_QNet(nn.Module):
    """Small MLP with optional normalisation and dropout.

    The size is driven by HIDDEN_SIZE and HIDDEN_LAYERS. Layer keys are l1, l2,
    l3 ... so checkpoints stay readable.
    """

    def __init__(self, input_size, hidden_size, output_size,
                 hidden_layers=HIDDEN_LAYERS, dropout_p=DROPOUT_P, use_norm=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.dropout_p = dropout_p
        self.use_norm = use_norm

        sizes = [input_size] + [hidden_size] * hidden_layers
        for i in range(hidden_layers):
            setattr(self, f"l{i + 1}", nn.Linear(sizes[i], sizes[i + 1]))
            if use_norm:
                setattr(self, f"ln{i + 1}", nn.LayerNorm(sizes[i + 1]))
        self.l4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        for i in range(self.hidden_layers):
            x = getattr(self, f"l{i + 1}")(x)
            if self.use_norm:
                x = getattr(self, f"ln{i + 1}")(x)
            x = F.relu(x)
            x = self.dropout(x)
        return self.l4(x)


class GridQNet(nn.Module):
    """Small conv net over the raw board, with global average pooling.

    Why this exists, stated plainly: the 11 feature state caps every learned
    agent at a mean of about 28 to 36, proven twice, once by 39 million DQN
    steps and once by imitation, where a student copied a teacher that scores
    132 and still only reached 28. The information needed to avoid traps is not
    in those 11 numbers.

    This network instead reads the whole board as channels: body, head, food,
    and the cell straight ahead. The global pooling makes the same weights read
    ANY board size, which is the whole point of the curriculum arms: train on
    10x10, play on 40x30, one set of weights, no retraining.

    Sized so it runs at useful speed on CPU: about 17k parameters, roughly the
    same as the old MLP. MaxPool(2) twice keeps the 40x30 board cheap; the
    smallest supported board is 8x8, where pooling leaves a 2x2 map for the
    global average to read.
    """

    def __init__(self, in_channels=4, output_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.output_size = output_size
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        # Global average pooling. This is what makes the network blind to board
        # size, which is what makes the curriculum transfer possible.
        x = x.mean(dim=(2, 3))
        return self.head(x)


def build_model(obs_mode, obs_width=None, output_size=3, use_norm=True,
                hidden_size=HIDDEN_SIZE, hidden_layers=HIDDEN_LAYERS):
    """One network factory for both trainers, so the width and the kind can
    never disagree with the environment that feeds them."""
    if obs_mode == "grid":
        return GridQNet(in_channels=4, output_size=output_size)
    return Linear_QNet(obs_width, hidden_size, output_size,
                       hidden_layers=hidden_layers, use_norm=use_norm)


class QTrainer:
    """Trains the Q network.

    Advanced method switches, all off by default so existing runs and
    checkpoints are unchanged:

      target           a lagged copy of the network computes the bootstrap
                       target. THE standard DQN fix. Without it the network
                       chases its own output every step.
      double_dqn       the online network picks the next action, the target
                       network scores it. Cuts the overestimation that plain
                       max() produces. Needs target=True to mean anything.
      n_step           the rewards in the replay buffer were already summed
                       over n steps by the caller. This only changes the
                       discount factor from gamma to gamma ** n.
    """

    def __init__(self, model, lr, gamma, weight_decay=WEIGHT_DECAY,
                 target=False, double_dqn=False, n_step=1, tau=0.01):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=weight_decay)
        # Huber loss is far less twitchy than MSE when a TD target overshoots.
        self.criterion = nn.SmoothL1Loss()

        self.use_target = bool(target)
        self.double_dqn = bool(double_dqn)
        self.n_step = max(1, int(n_step))
        self.tau = float(tau)
        self.target_model = None
        self.soft_updates = 0

        if self.use_target:
            import copy
            self.target_model = copy.deepcopy(model)
            self.target_model.eval()
            for p in self.target_model.parameters():
                p.requires_grad_(False)
            # Gradient clipping must not see the target copy twice, and its
            # parameters never need gradients.
            self._param_list = [p for p in model.parameters()]

        # Diagnostics, read by the logging layer. Left as None so that a caller
        # who does not log pays nothing for them. One float per step is cheap.
        self.last_loss = None
        self.last_grad_norm = None
        self.last_q_mean = None
        self.last_q_max = None
        self.last_target_mean = None

    def soft_update_target(self):
        """Polyak average the target towards the online network."""
        if self.target_model is None:
            return
        with torch.no_grad():
            for tp, op in zip(self.target_model.parameters(), self.model.parameters()):
                tp.mul_(1.0 - self.tau).add_(op, alpha=self.tau)
            for tb, ob in zip(self.target_model.buffers(), self.model.buffers()):
                tb.copy_(ob)
        self.soft_updates += 1

    def train_step(self, state, action, reward, next_state, done):
        device = next(self.model.parameters()).device

        state = torch.as_tensor(np.array(state), dtype=torch.float32, device=device)
        next_state = torch.as_tensor(np.array(next_state), dtype=torch.float32, device=device)
        reward = torch.as_tensor(np.array(reward), dtype=torch.float32, device=device)
        done = torch.as_tensor(np.array(done), dtype=torch.float32, device=device)

        # Single transition, unsqueeze into a batch of one so the code below has
        # exactly one shape to worry about.
        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        # action arrives as one hot vectors, so the chosen index is just the
        # argmax. Computed once for the whole batch.
        # A single step passes a 1D list like [1, 0, 0], so flatten first and
        # let argmax work on the last axis. Otherwise a 1D input has no dim=1
        # and the call raises IndexError.
        if torch.is_tensor(action):
            action_t = action.to(device)
        else:
            action_t = torch.as_tensor(np.array(action), dtype=torch.long, device=device)
        action_idx = action_t.reshape(-1, action_t.shape[-1]).argmax(dim=1)

        # 1: Predicted Q values with current state.
        pred = self.model(state)

        # 2: The bootstrap target. Computed in eval mode under no_grad so dropout
        # noise stays out of it and no graph is built.
        bootstrap_net = self.target_model if self.target_model is not None else self.model
        with torch.no_grad():
            was_training = self.model.training
            self.model.eval()
            bootstrap_net.eval()
            next_q = bootstrap_net(next_state)
            if self.double_dqn and self.target_model is not None:
                # Online net chooses the action, target net scores it.
                best_action = self.model(next_state).argmax(dim=1, keepdim=True)
                next_max = next_q.gather(1, best_action).squeeze(1)
            else:
                next_max = next_q.max(dim=1).values
            discount = self.gamma ** self.n_step
            target_q = reward + discount * next_max * (1.0 - done)
            if was_training:
                self.model.train()

        # Start from the predictions, then overwrite only the action that was
        # actually taken, which is what makes this a Q learning target.
        target = pred.clone()
        target[torch.arange(pred.size(0), device=device), action_idx] = target_q

        # 3: Calculate Loss and Backpropagate
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        # Gradient clipping keeps the occasional huge TD error from wrecking the
        # weights.
        params = self._param_list if self.use_target else self.model.parameters()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        self.optimizer.step()

        if self.target_model is not None:
            self.soft_update_target()

        # Record diagnostics without any extra forward or backward pass. A
        # collapsed grad norm here is the signal that learning has stalled, and
        # it is invisible if you only watch the score.
        self.last_loss = float(loss.detach())
        self.last_grad_norm = float(grad_norm)
        self.last_q_mean = float(pred.mean().detach())
        self.last_q_max = float(pred.max().detach())
        self.last_target_mean = float(target_q.mean().detach())

    def describe(self):
        bits = [f"lr={self.lr}", f"gamma={self.gamma}"]
        if self.use_target:
            bits.append(f"target_net(tau={self.tau})")
        if self.double_dqn and self.use_target:
            bits.append("double_dqn")
        if self.n_step > 1:
            bits.append(f"n_step={self.n_step}")
        return "QTrainer(" + ", ".join(bits) + ")"


# ==========================================
# CHECKPOINT HELPERS
# ==========================================

def architecture_fingerprint(input_size=11, output_size=3, use_norm=True):
    """Shape of the network, stored in the checkpoint and checked on load."""
    return {
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,
        "hidden_layers": HIDDEN_LAYERS,
        "output_size": output_size,
        "dropout_p": DROPOUT_P,
        # Recorded because a checkpoint from a norm net cannot load into a
        # no-norm net, and the mismatch should be reported rather than crash.
        "use_norm": bool(use_norm),
    }


def grid_architecture_fingerprint(in_channels=4, output_size=3):
    """Shape of a grid reading network, stored in the checkpoint and checked on
    load. Board size is deliberately NOT part of it: the global pooling makes
    the same weights legal on every board, which is the point of the curriculum
    arms."""
    return {
        "kind": "grid",
        "in_channels": in_channels,
        "output_size": output_size,
    }


def torch_load_checkpoint(path, device=None):
    """Load a checkpoint, accepting optimizer state.

    Modern PyTorch defaults weights_only=True, which rejects a checkpoint that
    carries optimizer state. Try the safe path first, then fall back.
    """
    device = device or DEVICE
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)
