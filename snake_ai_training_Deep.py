import pygame
import random
import numpy as np
from collections import deque
import torch
import os
import io

# Plotly is optional. Without it (or without kaleido) training still runs,
# it just skips the chart panel instead of dying on the first game over.
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Everything shared with the vectorized trainer lives in snake_core, which is
# pygame free on purpose so that script can run in an environment without
# pygame. See snake_core.py.
from snake_core import (  # noqa: E402
    BATCH_SIZE,
    BLOCK_SIZE,
    DEVICE,
    DROPOUT_P,
    EPSILON_DECAY_GAMES,
    EPSILON_MIN,
    EPSILON_START,
    GAMMA,
    HIDDEN_LAYERS,
    HIDDEN_SIZE,
    LR,
    MAX_MEMORY,
    REWARD_DEATH,
    REWARD_FOOD,
    SNAKE_THREADS,
    STARVATION_FRAMES_PER_SEGMENT,
    WEIGHT_DECAY,
    Direction,
    Linear_QNet,
    Point,
    QTrainer,
    architecture_fingerprint,
    set_threads,
    torch_load_checkpoint,
)

# This script trains one sample at a time, and one sample is the case where
# extra threads hurt. Measured: 1595 us per step on 1 thread against 2610 us on
# the default 16. The vectorized trainer sets its own default, because its
# batches are wide and may want the opposite.
THREADS = set_threads(SNAKE_THREADS if SNAKE_THREADS else 1)

# ==========================================
# RENDERING SETTINGS
# ==========================================
PLOT_UPDATE_EVERY = 5   # games between chart redraws, chart rendering is slow

# RGB Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
DARK_GRAY = (20, 20, 20)
PANEL_BG = (30, 30, 30)


class SnakeGameAI:
    def __init__(self, w=800, h=600):
        # Layout Dimensions
        self.left_w = 300   # Stats Panel
        self.game_w = w     # Game Area
        self.right_w = 700  # Plot Panel
        self.game_h = h

        self.total_w = self.left_w + self.game_w + self.right_w

        # Init Pygame
        pygame.init()
        self.display = pygame.display.set_mode((self.total_w, self.game_h))
        pygame.display.set_caption('Snake AI Training')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 20)
        self.title_font = pygame.font.SysFont('arial', 24, bold=True)
        self.score_font = pygame.font.SysFont('arial', 25)

        # Cached score label, rebuilt only when the score actually changes.
        self._score_cache_key = None
        self._score_surface = None

        self.plot_surface = None
        self.speed = 100
        self.game_history = []  # List of (game_num, score, record)
        self.scroll_offset = 0

        # Grid cell counts, used for integer coordinate maths and for food
        # placement on a board that can actually fill up.
        self.cols = self.game_w // BLOCK_SIZE
        self.rows = self.game_h // BLOCK_SIZE

        self.reset()

    def reset(self):
        # Init game state. Coordinates are kept as ints, floats silently break
        # Point equality and make collision checks unreliable.
        self.direction = Direction.RIGHT
        self.head = Point(self.cols // 2 * BLOCK_SIZE, self.rows // 2 * BLOCK_SIZE)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _all_cells(self):
        return [Point(c * BLOCK_SIZE, r * BLOCK_SIZE)
                for c in range(self.cols) for r in range(self.rows)]

    def _place_food(self):
        # Iterative, not recursive, so a nearly full board cannot blow the
        # Python stack. On a completely full board there is nowhere to put
        # food, and the episode should simply be over.
        occupied = set(self.snake)
        free = [c for c in self._all_cells() if c not in occupied]
        if not free:
            self.food = None
            return
        self.food = random.choice(free)

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. Collect user input (to allow quitting and UI interaction)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()

                # Speed Bar Interaction (Bottom Right)
                speed_bar_x = self.left_w + self.game_w + 20
                speed_bar_y = self.game_h - 40
                speed_bar_w = self.right_w - 40

                if speed_bar_x <= mx <= speed_bar_x + speed_bar_w and speed_bar_y <= my <= speed_bar_y + 20:
                    ratio = (mx - speed_bar_x) / speed_bar_w
                    ratio = max(0, min(1, ratio))
                    self.speed = int(ratio * 500) + 1  # 1 to 501

            elif event.type == pygame.MOUSEWHEEL:
                # Scroll Stats (Left Panel)
                mx, my = pygame.mouse.get_pos()
                if 0 <= mx <= self.left_w:
                    self.scroll_offset -= event.y
                    self.scroll_offset = max(0, self.scroll_offset)

        # 2. Move
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. Check if game over
        reward = 0.0
        # The board is full: the snake won, there is nothing left to eat.
        if self.food is None:
            return REWARD_FOOD, True, self.score

        # If collision or loop takes too long (starvation guard)
        if self.is_collision() or self.frame_iteration > STARVATION_FRAMES_PER_SEGMENT * len(self.snake):
            return REWARD_DEATH, True, self.score

        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = REWARD_FOOD
            self._place_food()
        else:
            self.snake.pop()

        # 5. Update UI and Clock
        self._update_ui()
        self.clock.tick(self.speed)
        return reward, False, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x > self.game_w - BLOCK_SIZE or pt.x < 0 or pt.y > self.game_h - BLOCK_SIZE or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        # --- Draw Left Panel (Stats) ---
        self.draw_stats()

        # --- Draw Center (Game) ---
        # Draw game background offset by self.left_w
        game_rect = pygame.Rect(self.left_w, 0, self.game_w, self.game_h)
        pygame.draw.rect(self.display, BLACK, game_rect)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(self.left_w + pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(self.left_w + pt.x + 4, pt.y + 4, 12, 12))

        if self.food is not None:
            pygame.draw.rect(self.display, RED, pygame.Rect(self.left_w + self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw Score in Game Area, cached per score value instead of rebuilding
        # a font surface every single frame.
        if self._score_cache_key != self.score:
            self._score_surface = self.score_font.render("Score: " + str(self.score), True, WHITE)
            self._score_cache_key = self.score
        self.display.blit(self._score_surface, [self.left_w + 10, 10])

        # --- Draw Right Panel (Plot & Speed) ---
        pygame.draw.rect(self.display, DARK_GRAY, pygame.Rect(self.left_w + self.game_w, 0, self.right_w, self.game_h))

        if self.plot_surface:
            self.display.blit(self.plot_surface, (self.left_w + self.game_w, 0))
        elif not PLOTLY_AVAILABLE:
            msg = self.font.render("Chart disabled (plotly missing)", True, GRAY)
            self.display.blit(msg, (self.left_w + self.game_w + 20, 20))

        self.draw_speed_bar()

        pygame.display.flip()

    def draw_stats(self):
        start_x = 10
        start_y = 10
        width = self.left_w - 20
        height = self.game_h - 20

        # Background
        pygame.draw.rect(self.display, PANEL_BG, pygame.Rect(start_x, start_y, width, height))

        # Header
        header_text = self.title_font.render("Game History", True, WHITE)
        self.display.blit(header_text, (start_x + 10, start_y + 10))

        # Column Headers
        headers = ["Game", "Score", "Record"]
        col_widths = [70, 70, 70]
        curr_x = start_x + 10
        y_offset = start_y + 45

        for i, h in enumerate(headers):
            text = self.font.render(h, True, YELLOW)
            self.display.blit(text, (curr_x, y_offset))
            curr_x += col_widths[i]

        # List Items (Scrollable)
        item_y = y_offset + 30
        visible_items = (height - 80) // 25

        max_scroll = max(0, len(self.game_history) - visible_items)
        self.scroll_offset = min(self.scroll_offset, max_scroll)

        start_idx = max(0, len(self.game_history) - visible_items - self.scroll_offset)
        end_idx = len(self.game_history) - self.scroll_offset

        subset = self.game_history[start_idx:end_idx] if self.game_history else []

        for game_num, score, record in subset:
            curr_x = start_x + 10

            # Game Num
            text = self.font.render(f"{game_num}", True, WHITE)
            self.display.blit(text, (curr_x, item_y))
            curr_x += col_widths[0]

            # Score
            text = self.font.render(str(score), True, GREEN)
            self.display.blit(text, (curr_x, item_y))
            curr_x += col_widths[1]

            # Record
            text = self.font.render(str(record), True, BLUE2)
            self.display.blit(text, (curr_x, item_y))

            item_y += 25

    def draw_speed_bar(self):
        start_x = self.left_w + self.game_w + 20
        start_y = self.game_h - 40
        width = self.right_w - 40
        height = 10

        # Label
        label = self.font.render(f"Speed: {self.speed}", True, WHITE)
        self.display.blit(label, (start_x, start_y - 25))

        # Bar Background
        pygame.draw.rect(self.display, GRAY, pygame.Rect(start_x, start_y, width, height))

        # Handle
        ratio = (self.speed - 1) / 500
        handle_x = start_x + int(ratio * width)
        pygame.draw.circle(self.display, RED, (handle_x, start_y + 5), 8)

    def update_plot_data(self, scores, mean_scores):
        if not PLOTLY_AVAILABLE:
            return
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=scores, mode='lines', name='Score', line=dict(color='#00bfff')))
            fig.add_trace(go.Scatter(y=mean_scores, mode='lines', name='Mean', line=dict(color='#ff4500')))

            fig.update_layout(
                template='plotly_dark',
                title=dict(text='Training Progress', font=dict(size=16, color='white')),
                xaxis_title='Games',
                yaxis_title='Score',
                width=self.right_w,
                height=self.game_h - 100,
                margin=dict(l=50, r=20, t=50, b=50),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30,30,30,1)'
            )

            # Convert to image bytes, then load into Pygame. Kaleido is what
            # actually renders this, so a missing kaleido is caught here rather
            # than crashing the training run.
            img_bytes = pio.to_image(fig, format="png")
            self.plot_surface = pygame.image.load(io.BytesIO(img_bytes))
        except Exception as e:
            # Only warn once per process, this is called every few games.
            if not getattr(self, "_plot_warned", False):
                print(f"=> Chart rendering disabled ({type(e).__name__}: {e})")
                print("=> Install kaleido to get the live plot panel back.")
                self._plot_warned = True
            self.plot_surface = None

    def _move(self, action):
        # Action is [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        # int() keeps Point equality reliable.
        self.head = Point(int(x), int(y))

# ==========================================
# PART 2: THE BRAIN
# ==========================================
# Linear_QNet and QTrainer live in snake_core.py now, which carries no pygame
# dependency. They are imported at the top of this file.

# ==========================================
# PART 3: THE AGENT (Orchestrator)
# ==========================================

CHECKPOINT_PATH = './model/deep_checkpoint.pth'


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = EPSILON_START
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)

        # Initialize model and move to DEVICE
        self.model = Linear_QNet(
            input_size=11,
            hidden_size=HIDDEN_SIZE,
            output_size=3,
            hidden_layers=HIDDEN_LAYERS,
            dropout_p=DROPOUT_P,
        ).to(DEVICE)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        # Resume state
        self.record = 0
        self.total_score = 0
        self.load_checkpoint()

    # Architecture fingerprint, so a checkpoint from a different shaped network
    # is rejected loudly instead of half loading. Shared with the vectorized
    # trainer, so both agree on what a matching checkpoint looks like.
    def _architecture(self):
        return architecture_fingerprint()

    def load_checkpoint(self):
        if not os.path.exists(CHECKPOINT_PATH):
            print("=> No checkpoint found. Starting fresh.")
            return

        print("=> Loading checkpoint...")
        try:
            checkpoint = torch_load_checkpoint(CHECKPOINT_PATH, DEVICE)

            saved_arch = checkpoint.get('architecture')
            if saved_arch is not None and saved_arch != self._architecture():
                print("=> Checkpoint was saved with a different network shape:")
                print(f"   saved : {saved_arch}")
                print(f"   current: {self._architecture()}")
                print("=> Starting fresh. The old file is left untouched.")
                return

            self.model.load_state_dict(checkpoint['model_state'])
            try:
                self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
            except (ValueError, KeyError) as e:
                # Optimizer state (AdamW moments) is shape bound too. Model
                # weights survived, so keep them and just reset the optimiser.
                print(f"=> Optimizer state did not match ({e}).")
                print("=> Keeping the model weights, restarting the optimizer.")

            self.n_games = checkpoint['n_games']
            self.record = checkpoint['record']
            self.total_score = checkpoint.get('total_score', 0)
            print(f"=> Resumed from game {self.n_games} with record {self.record}")
        except Exception as e:
            print(f"=> Could not load checkpoint: {type(e).__name__}: {e}")
            print("=> Starting fresh.")

    def save_checkpoint(self):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        checkpoint = {
            'n_games': self.n_games,
            'record': self.record,
            'total_score': self.total_score,
            'architecture': self._architecture(),
            'model_state': self.model.state_dict(),
            'optimizer_state': self.trainer.optimizer.state_dict()
        }
        torch.save(checkpoint, CHECKPOINT_PATH)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food is not None and game.food.x < game.head.x,  # food left
            game.food is not None and game.food.x > game.head.x,  # food right
            game.food is not None and game.food.y < game.head.y,  # food up
            game.food is not None and game.food.y > game.head.y   # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random moves: tradeoff exploration / exploitation.
        # Epsilon decays multiplicatively with a floor, so late training keeps a
        # small amount of exploration instead of collapsing to pure greed.
        progress = min(1.0, self.n_games / EPSILON_DECAY_GAMES)
        self.epsilon = max(EPSILON_MIN, EPSILON_START * ((1.0 - progress) ** 1.5))

        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randrange(3)
        else:
            with torch.no_grad():
                state0 = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
                prediction = self.model(state0)
                move = torch.argmax(prediction).item()

        final_move[move] = 1
        return final_move

# ==========================================
# PART 4: MAIN EXECUTION LOOP
# ==========================================

def save_final_plot(scores, mean_scores, path='./model/training_plot.png'):
    if not PLOTLY_AVAILABLE or not scores:
        return
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=scores, mode='lines', name='Score'))
        fig.add_trace(go.Scatter(y=mean_scores, mode='lines', name='Mean'))
        fig.update_layout(template='plotly_dark', title='Training Progress',
                          xaxis_title='Games', yaxis_title='Score')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.write_image(path)
        print(f"=> Saved training chart to {path}")
    except Exception as e:
        print(f"=> Could not save the training chart ({type(e).__name__}: {e})")


def train():
    plot_scores = []
    plot_mean_scores = []
    agent = Agent()
    game = SnakeGameAI()

    # Load initial state from agent (in case of resume)
    total_score = agent.total_score
    record = agent.record

    print("Starting Training... (Press Ctrl+C to stop)")

    try:
        while True:
            # 1. Get old state
            state_old = agent.get_state(game)

            # 2. Get move
            final_move = agent.get_action(state_old)

            # 3. Perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # 4. Train short memory (for specific step)
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # 5. Remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # Train long memory (Experience Replay), plot result
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.record = record
                    agent.save_checkpoint()

                print(f'Game {agent.n_games} Score {score} Record {record} Epsilon {agent.epsilon:.3f}')

                plot_scores.append(score)
                total_score += score
                agent.total_score = total_score

                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)

                # Update UI Data. Chart rendering is expensive, so it is
                # throttled rather than redrawn on every single game.
                game.game_history.append((agent.n_games, score, record))
                if agent.n_games % PLOT_UPDATE_EVERY == 0:
                    game.update_plot_data(plot_scores, plot_mean_scores)

                # Save checkpoint every 10 games just in case record isn't broken
                if agent.n_games % 10 == 0:
                    agent.save_checkpoint()

    except KeyboardInterrupt:
        print("Training stopped by user. Saving final state...")
        agent.save_checkpoint()
        save_final_plot(plot_scores, plot_mean_scores)
        pygame.quit()


if __name__ == '__main__':
    train()
