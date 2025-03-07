"""
Taxi-v3 Reinforcement Learning Implementation
Author: Malak Zaidi
Date: March 2025
Description: Implementation of Q-learning, SARSA, and DQN for the OpenAI Gym Taxi-v3 environment.
"""

import time
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from itertools import product
import torch
import torch.nn as nn
import torch.optim as optim

class RLAgent:
    def __init__(self, env_name='Taxi-v3'):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.render_env = gym.make(env_name, render_mode="human")

        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            self.state_space = self.env.observation_space.n
            self.discrete_state = True
        else:
            self.state_space = self.env.observation_space.shape[0]
            self.discrete_state = False

        self.action_space = self.env.action_space.n
        self.Q = np.zeros((self.state_space, self.action_space)) if self.discrete_state else None

        self.learning_params = {
            'alpha': 0.8,
            'gamma': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'min_epsilon': 0.01,
        }

        self.training_rewards = []
        self.episode_lengths = []

    def train_q_learning(self, episodes=1000):
        print("\nTraining Q-Learning Agent...")
        rewards = []
        for episode in tqdm(range(episodes)):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                if np.random.rand() < self.learning_params['epsilon']:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Q[state, :])

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                best_next_action = np.argmax(self.Q[next_state, :])
                td_target = reward + self.learning_params['gamma'] * self.Q[next_state, best_next_action]
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.learning_params['alpha'] * td_error

                state = next_state
                total_reward += reward

            self.learning_params['epsilon'] = max(
                self.learning_params['min_epsilon'],
                self.learning_params['epsilon'] * self.learning_params['epsilon_decay']
            )
            rewards.append(total_reward)

        self.training_rewards = rewards
        print("\nTraining Completed!")
        print(f"Average Reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")

    def train_sarsa(self, episodes=1000):
        print("\nTraining SARSA Agent...")
        rewards = []
        for episode in tqdm(range(episodes)):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            if np.random.rand() < self.learning_params['epsilon']:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(self.Q[state, :])

            while not done:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                if np.random.rand() < self.learning_params['epsilon']:
                    next_action = self.env.action_space.sample()
                else:
                    next_action = np.argmax(self.Q[next_state, :])

                td_target = reward + self.learning_params['gamma'] * self.Q[next_state, next_action]
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.learning_params['alpha'] * td_error

                state, action = next_state, next_action
                total_reward += reward

            self.learning_params['epsilon'] = max(
                self.learning_params['min_epsilon'],
                self.learning_params['epsilon'] * self.learning_params['epsilon_decay']
            )
            rewards.append(total_reward)

        self.training_rewards = rewards
        print("\nTraining Completed!")
        print(f"Average Reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")

    def initialize_dqn(self):
        """Initialize a Deep Q-Network (DQN) using PyTorch."""
        self.dqn_model = nn.Sequential(
            nn.Linear(self.state_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space)
        )
        self.target_model = nn.Sequential(
            nn.Linear(self.state_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space)
        )
        self.target_model.load_state_dict(self.dqn_model.state_dict())
        self.optimizer = optim.Adam(self.dqn_model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def train_dqn(self, episodes=1000, batch_size=32, target_update_freq=10):
        print("\nTraining DQN Agent...")
        rewards = []
        replay_buffer = []

        for episode in tqdm(range(episodes)):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                if np.random.rand() < self.learning_params['epsilon']:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state)
                        q_values = self.dqn_model(state_tensor)
                        action = torch.argmax(q_values).item()

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > 10000:
                    replay_buffer.pop(0)

                if len(replay_buffer) >= batch_size:
                    batch = np.random.choice(replay_buffer, batch_size, replace=False)
                    states = torch.FloatTensor(np.array([exp[0] for exp in batch]))
                    actions = torch.LongTensor(np.array([exp[1] for exp in batch]))
                    rewards_batch = torch.FloatTensor(np.array([exp[2] for exp in batch]))
                    next_states = torch.FloatTensor(np.array([exp[3] for exp in batch]))
                    dones = torch.FloatTensor(np.array([exp[4] for exp in batch]))

                    with torch.no_grad():
                        target_q_values = self.target_model(next_states)
                        targets = rewards_batch + self.learning_params['gamma'] * torch.max(target_q_values, dim=1).values * (1 - dones)

                    q_values = self.dqn_model(states)
                    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                    loss = self.loss_fn(q_values, targets)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                state = next_state
                total_reward += reward

            if episode % target_update_freq == 0:
                self.target_model.load_state_dict(self.dqn_model.state_dict())

            self.learning_params['epsilon'] = max(
                self.learning_params['min_epsilon'],
                self.learning_params['epsilon'] * self.learning_params['epsilon_decay']
            )
            rewards.append(total_reward)

        self.training_rewards = rewards
        print("\nTraining Completed!")
        print(f"Average Reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")

    def compare_algorithms(self, env_name='Taxi-v3', algorithms=['q_learning', 'sarsa', 'dqn'], episodes=500):
        results = {}

        for algo in algorithms:
            self.__init__(env_name)
            self.learning_params['epsilon'] = 1.0

            if algo == 'q_learning':
                avg_reward = self.train_q_learning(episodes=episodes)
            elif algo == 'sarsa':
                avg_reward = self.train_sarsa(episodes=episodes)
            elif algo == 'dqn':
                self.initialize_dqn()
                avg_reward = self.train_dqn(episodes=episodes)

            results[algo] = {
                'avg_reward': avg_reward,
                'training_rewards': self.training_rewards.copy(),
            }

        self.plot_algorithm_comparison(results)
        return results

    def plot_algorithm_comparison(self, results):
        plt.figure(figsize=(12, 6))
        for algo, data in results.items():
            plt.plot(data['training_rewards'], label=f'{algo} (Avg: {data["avg_reward"]:.2f})')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Algorithm Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()

    def explain_policy(self, state):
        if self.discrete_state:
            q_values = self.Q[state, :]
            best_action = np.argmax(q_values)
            action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]
            explanation = f"Best Action: {action_names[best_action]}\n"
            explanation += "Q-values:\n"
            for i, q_val in enumerate(q_values):
                explanation += f"- {action_names[i]}: {q_val:.2f}\n"
            return explanation
        else:
            return "Policy explanation is not available for continuous state spaces."

    def experiment_with_environments(self, env_names):
        results = {}
        for env_name in env_names:
            print(f"\nExperimenting with {env_name}...")
            self.__init__(env_name)
            self.train_q_learning(episodes=1000)
            results[env_name] = np.mean(self.training_rewards[-100:])
        print("\nExperiment Results:")
        for env, reward in results.items():
            print(f"{env}: Average Reward = {reward:.2f}")

if __name__ == "__main__":
    agent = RLAgent(env_name='Taxi-v3')
    agent.compare_algorithms(episodes=500)
    agent.experiment_with_environments(['Taxi-v3', 'FrozenLake-v1', 'CartPole-v1'])