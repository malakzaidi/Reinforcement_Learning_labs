import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

# Define the parameters
GRID_SIZE = 4
STATE_SIZE = GRID_SIZE * GRID_SIZE
ACTION_SIZE = 4  # Up, Down, Left, Right
GAMMA = 0.9  # Discount factor
LEARNING_RATE = 0.01
EPSILON = 1.0  # Initial exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 2000
EPISODES = 1000

# Define possible moves
MOVES = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}

# Define the GridWorld environment
class GridWorld:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.reset()

    def reset(self):
        """Reset the agent to the starting position."""
        self.agent_pos = (0, 0)
        self.goal_pos = (3, 3)
        self.obstacle_pos = (1, 1)  # Obstacle position
        return self.get_state()

    def get_state(self):
        """Return the state as a binary vector."""
        state = np.zeros((self.grid_size, self.grid_size))
        state[self.agent_pos] = 1
        return state.flatten()

    def step(self, action):
        """Move the agent and return the new state, reward, and done status."""
        x, y = self.agent_pos
        dx, dy = MOVES[action]
        new_x, new_y = x + dx, y + dy

        # Check boundaries
        if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
            self.agent_pos = (new_x, new_y)

        # Check reward
        if self.agent_pos == self.goal_pos:
            return self.get_state(), 10, True  # Goal reached
        elif self.agent_pos == self.obstacle_pos:
            return self.get_state(), -5, False  # Obstacle hit
        else:
            return self.get_state(), -1, False  # Normal move

# Define the DQN Agent
class DQNAgent:
    def __init__(self):
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.model = self._build_model()

    def _build_model(self):
        """Build the neural network model."""
        model = Sequential([
            Dense(24, activation='relu', input_shape=(self.state_size,)),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store an experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose an action using epsilon-greedy strategy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Exploration
        q_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])  # Exploitation

    def replay(self):
        """Train the model with past experiences."""
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            target = self.model.predict(np.array([state]), verbose=0)[0]
            if done:
                target[action] = reward
            else:
                target[action] = reward + GAMMA * np.max(self.model.predict(np.array([next_state]), verbose=0)[0])
            self.model.fit(np.array([state]), np.array([target]), epochs=1, verbose=0)
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY  # Decay exploration rate

# Training the agent
env = GridWorld()
agent = DQNAgent()

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    for step in range(50):  # Limit to 50 moves
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
    agent.replay()
    print(f"Episode {episode+1}/{EPISODES}, Score: {total_reward}, Epsilon: {agent.epsilon:.4f}")

# Save the model
agent.model.save("dqn_model.keras")