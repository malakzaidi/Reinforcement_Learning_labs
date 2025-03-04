import gymnasium as gym
import numpy as np
import pygame
import time
import matplotlib.pyplot as plt

# Initialize Gymnasium environment
env = gym.make('Taxi-v3', render_mode="human")
while hasattr(env, 'env'):
    env = env.env

# PyGame settings
pygame.init()
screen_width, screen_height = 500, 500
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("AI Taxi - Q-Learning Visualization")
clock = pygame.time.Clock()

# Colors
WHITE, BLACK, GRAY = (255, 255, 255), (0, 0, 0), (40, 40, 40)
YELLOW, RED, GREEN, BLUE = (255, 223, 0), (200, 0, 0), (0, 200, 0), (0, 100, 255)

# Grid settings
grid_size = 5
cell_size = screen_width // grid_size
font = pygame.font.Font(None, 28)

# Load images
taxi_img = pygame.image.load("taxi.png")
taxi_img = pygame.transform.scale(taxi_img, (cell_size, cell_size))
passenger_img = pygame.image.load("passenger.png")
passenger_img = pygame.transform.scale(passenger_img, (cell_size // 2, cell_size // 2))
destination_img = pygame.image.load("destination.png")
destination_img = pygame.transform.scale(destination_img, (cell_size // 2, cell_size // 2))


def draw_grid():
    for x in range(0, screen_width, cell_size):
        for y in range(0, screen_height, cell_size):
            rect = pygame.Rect(x, y, cell_size, cell_size)
            pygame.draw.rect(screen, GRAY, rect, 1)


def render_environment(state, episode, total_reward):
    screen.fill(BLACK)
    draw_grid()
    taxi_row, taxi_col, pass_idx, dest_idx = env.decode(state)
    screen.blit(taxi_img, (taxi_col * cell_size, taxi_row * cell_size))
    if pass_idx < 4:
        pass_row, pass_col = env.locs[pass_idx]
        screen.blit(passenger_img, (pass_col * cell_size + 10, pass_row * cell_size + 10))
    else:
        screen.blit(passenger_img, (taxi_col * cell_size + 10, taxi_row * cell_size + 10))
    dest_row, dest_col = env.locs[dest_idx]
    screen.blit(destination_img, (dest_col * cell_size + 10, dest_row * cell_size + 10))
    info_text = font.render(f"Episode: {episode}  Reward: {total_reward}", True, WHITE)
    screen.blit(info_text, (10, 10))
    pygame.display.flip()


# Q-Learning setup
state_space, action_space = env.observation_space.n, env.action_space.n
Q = np.zeros((state_space, action_space))

# Optimized Hyperparameters
alpha, gamma = 0.5, 0.99  # Faster learning
epsilon, epsilon_decay, min_epsilon = 1.0, 0.995, 0.01  # More aggressive decay
episodes = 5000
render_interval = 200  # Less frequent rendering

rewards_per_episode, steps_per_episode, avg_q_values = [], [], []

print("\nTraining the AI Taxi...\n")
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward, step_count = 0, 0

    while not done:
        action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[state, :])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
        state = next_state
        total_reward += reward
        step_count += 1

        if episode % render_interval == 0:
            render_environment(state, episode, total_reward)
            clock.tick(20)

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards_per_episode.append(total_reward)
    steps_per_episode.append(step_count)
    avg_q_values.append(np.mean(Q))

    if episode % 500 == 0:
        print(f"Episode: {episode}, Reward: {total_reward}, Steps: {step_count}, Epsilon: {epsilon:.4f}")

print("\nTraining Complete! 🎉")

# Graphs for better analysis
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(rewards_per_episode, label="Rewards", color='b')
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Rewards Over Episodes")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(steps_per_episode, label="Steps Taken", color='r')
plt.xlabel("Episodes")
plt.ylabel("Steps")
plt.title("Steps Taken Over Episodes")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(avg_q_values, label="Average Q-Value", color='g')
plt.xlabel("Episodes")
plt.ylabel("Q-Value")
plt.title("Q-Value Evolution")
plt.legend()

plt.tight_layout()
plt.show()

print("\n🚀 Testing Trained Model...\n")
state, _ = env.reset()
done, total_reward, step_count = False, 0, 0
time.sleep(2)

while not done:
    action = np.argmax(Q[state, :])
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = next_state
    total_reward += reward
    step_count += 1
    render_environment(state, "Test", total_reward)
    clock.tick(10)

print(f"\n✅ Test Complete! Total Reward: {total_reward}, Steps Taken: {step_count}")

env.close()
pygame.quit()
