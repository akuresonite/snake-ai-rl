import pygame
import random
import numpy as np
from collections import deque
from enum import Enum
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

# Try importing Intel Extension for PyTorch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    ipex = None

# ==========================================
# DEVICE CONFIGURATION
# ==========================================
def get_device():
    # Check for CUDA (Nvidia)
    if torch.cuda.is_available():
        print(f"Using CUDA Device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    
    # Check for XPU (Intel)
    try:
        if ipex and hasattr(torch, 'xpu') and torch.xpu.is_available():
            print(f"Using Intel XPU: {torch.xpu.get_device_name(0)}")
            return torch.device("xpu")
    except Exception as e:
        pass

    print("Using CPU")
    return torch.device("cpu")

DEVICE = get_device()

# ==========================================
# PART 1: GAME ENVIRONMENT (The World)
# ==========================================

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# RGB Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40  # Increase this to speed up training visualization

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # Init Pygame
        pygame.init()
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI Training')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Init game state
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. Collect user input (to allow quitting)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Move
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. Check if game over
        reward = 0
        game_over = False
        # If collision or loop takes too long (starvation guard)
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. Update UI and Clock
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = pygame.font.SysFont('arial', 25).render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # Action is [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn

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

        self.head = Point(x, y)

# ==========================================
# PART 2: THE BRAIN (Neural Network)
# ==========================================

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Move tensors to the configured DEVICE
        state = torch.tensor(np.array(state), dtype=torch.float).to(DEVICE)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(DEVICE)
        action = torch.tensor(np.array(action), dtype=torch.long).to(DEVICE)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(DEVICE)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x) -> Appending 1 dimension for single step training
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: Predicted Q values with current state
        pred = self.model(state)

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 3: Calculate Loss and Backpropagate
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

# ==========================================
# PART 3: THE AGENT (Orchestrator)
# ==========================================

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
CHECKPOINT_PATH = './model/checkpoint.pth'

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY) 
        
        # Initialize model and move to DEVICE
        self.model = Linear_QNet(11, 256, 3).to(DEVICE)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # Resume state
        self.record = 0
        self.total_score = 0
        self.load_checkpoint()

    def load_checkpoint(self):
        if os.path.exists(CHECKPOINT_PATH):
            print("=> Loading checkpoint...")
            try:
                # Load to CPU first to avoid device mapping issues, then move
                checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
                self.model.load_state_dict(checkpoint['model_state'])
                self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
                self.n_games = checkpoint['n_games']
                self.record = checkpoint['record']
                self.total_score = checkpoint.get('total_score', 0)
                print(f"=> Resumed from game {self.n_games} with record {self.record}")
            except Exception as e:
                print(f"=> Error loading checkpoint: {e}. Starting fresh.")
        else:
            print("=> No checkpoint found. Starting fresh.")

    def save_checkpoint(self):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        checkpoint = {
            'n_games': self.n_games,
            'record': self.record,
            'total_score': self.total_score,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.trainer.optimizer.state_dict()
        }
        torch.save(checkpoint, CHECKPOINT_PATH)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

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
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
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
        # Random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Convert to tensor and move to DEVICE
            state0 = torch.tensor(state, dtype=torch.float).to(DEVICE)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

# ==========================================
# PART 4: PLOTTING HELPER
# ==========================================

def plot(scores, mean_scores):
    # Initialize plot objects if they don't exist
    if not hasattr(plot, "fig"):
        plt.ion()
        plot.fig = plt.figure()
        plot.ax = plot.fig.add_subplot(111)
        plot.line1, = plot.ax.plot([], [], label='Score')
        plot.line2, = plot.ax.plot([], [], label='Mean Score')
        plot.ax.set_title('Training...')
        plot.ax.set_xlabel('Number of Games')
        plot.ax.set_ylabel('Score')
        plot.ax.legend()
        plot.text_score = plot.ax.text(0, 0, '')
        plot.text_mean = plot.ax.text(0, 0, '')
        plt.show(block=False)

    try:
        # Update lines
        plot.line1.set_data(range(len(scores)), scores)
        plot.line2.set_data(range(len(mean_scores)), mean_scores)
        
        # Update text
        if scores:
            plot.text_score.set_position((len(scores)-1, scores[-1]))
            plot.text_score.set_text(str(scores[-1]))
            plot.text_mean.set_position((len(mean_scores)-1, mean_scores[-1]))
            plot.text_mean.set_text(str(mean_scores[-1]))

        # Rescale view
        plot.ax.relim()
        plot.ax.autoscale_view()
        
        # Explicitly set Y-limits to avoid microscopic range (e.g. 0 to 0.05)
        # when all scores are 0.
        ymax = max(max(scores) if scores else 0, max(mean_scores) if mean_scores else 0)
        # Ensure at least a range of 0 to 1
        if ymax < 1:
            ymax = 1
        plot.ax.set_ylim(bottom=0, top=ymax + 1) # Add slight padding

        # Redraw
        plot.fig.canvas.draw()
        plot.fig.canvas.flush_events()
        
    except Exception:
        pass

# ==========================================
# PART 5: MAIN EXECUTION LOOP
# ==========================================

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

                print(f'Game {agent.n_games} Score {score} Record {record}')

                plot_scores.append(score)
                total_score += score
                agent.total_score = total_score
                
                # Avoid division by zero if n_games is 0 (shouldn't happen here)
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)
                
                # Save checkpoint every 10 games just in case record isn't broken
                if agent.n_games % 10 == 0:
                    agent.save_checkpoint()

    except KeyboardInterrupt:
        print("Training stopped by user. Saving final state...")
        agent.save_checkpoint()
        pygame.quit()

if __name__ == '__main__':
    # Turn on interactive mode for Matplotlib
    plt.ion()
    train()