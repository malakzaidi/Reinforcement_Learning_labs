import gymnasium as gym
import numpy as np
import pygame
import time
import matplotlib.pyplot as plt
import random
import sys

import sys

sys.path.append("/usr/local/lib/python3.10/dist-packages")

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


class AdvancedTaxiEnvironment:
    def __init__(self, screen_width: int = 1200, screen_height: int = 800):
        # Environment setup
        self.env = gym.make('Taxi-v3', render_mode="human")
        while hasattr(self.env, 'env'):
            self.env = self.env.env

        # Advanced simulation parameters
        self.traffic_scenarios = [
            {'name': 'Light Traffic', 'density': 0.1, 'speed_multiplier': 1.0},
            {'name': 'Medium Traffic', 'density': 0.3, 'speed_multiplier': 0.8},
            {'name': 'Heavy Traffic', 'density': 0.5, 'speed_multiplier': 0.6}
        ]

        # Dynamic weather conditions
        self.weather_conditions = [
            {'name': 'Clear', 'visibility': 1.0, 'difficulty_multiplier': 1.0},
            {'name': 'Rainy', 'visibility': 0.7, 'difficulty_multiplier': 1.3},
            {'name': 'Foggy', 'visibility': 0.4, 'difficulty_multiplier': 1.5}
        ]

        # Passenger profiles
        self.passenger_profiles = [
            {
                'type': 'Business Traveler',
                'urgency_level': 0.9,
                'fare_multiplier': 1.5,
                'tip_probability': 0.7
            },
            {
                'type': 'Tourist',
                'urgency_level': 0.5,
                'fare_multiplier': 1.0,
                'tip_probability': 0.4
            },
            {
                'type': 'Student',
                'urgency_level': 0.3,
                'fare_multiplier': 0.8,
                'tip_probability': 0.2
            }
        ]

        # Pygame setup
        pygame.init()
        self.screen_width, self.screen_height = screen_width, screen_height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Advanced Autonomous Taxi Simulator")

        # Colors
        self.COLORS = {
            'BACKGROUND': (20, 20, 30),
            'GRID': (50, 50, 70),
            'TAXI': (255, 200, 0),
            'PASSENGER': (0, 255, 100),
            'DESTINATION': (255, 50, 50),
            'TEXT': (255, 255, 255)
        }

        # Fonts
        self.large_font = pygame.font.Font(None, 48)
        self.medium_font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Simulation tracking
        self.performance_metrics = {
            'total_trips': 0,
            'total_earnings': 0,
            'total_tips': 0,
            'successful_deliveries': 0,
            'episode_rewards': [],
            'episode_steps': []
        }

        # Q-Learning parameters
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.learning_params = {
            'alpha': 0.5,  # Learning rate
            'gamma': 0.99,  # Discount factor
            'epsilon': 1.0,  # Exploration rate
            'epsilon_decay': 0.995,
            'min_epsilon': 0.01
        }

    def generate_dynamic_scenario(self):
        """Generate a dynamic scenario for each episode."""
        return {
            'traffic': random.choice(self.traffic_scenarios),
            'weather': random.choice(self.weather_conditions),
            'passenger': random.choice(self.passenger_profiles)
        }

    def train(self, episodes=500):
        """Advanced training method with dynamic scenarios."""
        print("\n🚀 Autonomous Taxi Training Started 🚀")

        for episode in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0

            # Generate dynamic scenario
            scenario = self.generate_dynamic_scenario()

            while not done:
                # Action selection
                if random.random() < self.learning_params['epsilon']:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Q[state, :])

                # Step through environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Modify reward based on scenario
                adjusted_reward = reward * scenario['passenger']['fare_multiplier'] * \
                                  scenario['weather']['difficulty_multiplier']

                # Q-learning update
                self.Q[state, action] += self.learning_params['alpha'] * (
                        adjusted_reward +
                        self.learning_params['gamma'] * np.max(self.Q[next_state, :]) -
                        self.Q[state, action]
                )

                state = next_state
                total_reward += adjusted_reward
                steps += 1

            # Performance tracking
            self.performance_metrics['episode_rewards'].append(total_reward)
            self.performance_metrics['episode_steps'].append(steps)

            # Decay exploration
            self.learning_params['epsilon'] = max(
                self.learning_params['min_epsilon'],
                self.learning_params['epsilon'] * self.learning_params['epsilon_decay']
            )

            # Progress report
            if episode % 50 == 0:
                print(f"Episode {episode}: "
                      f"Reward = {total_reward:.2f}, "
                      f"Steps = {steps}, "
                      f"Scenario = {scenario['traffic']['name']}/{scenario['weather']['name']}")

        print("\n✅ Training Completed Successfully!")

    def visualize_performance_plotly(self):
        """Create interactive visualizations using Plotly."""
        # Rewards over Episodes
        fig_rewards = go.Figure(data=go.Scatter(
            y=self.performance_metrics['episode_rewards'],
            mode='lines',
            name='Rewards',
            line=dict(color='blue', width=2)
        ))
        fig_rewards.update_layout(
            title='Rewards per Episode',
            xaxis_title='Episode',
            yaxis_title='Reward'
        )
        fig_rewards.show()

        # Steps per Episode
        fig_steps = go.Figure(data=go.Scatter(
            y=self.performance_metrics['episode_steps'],
            mode='lines',
            name='Steps',
            line=dict(color='green', width=2)
        ))
        fig_steps.update_layout(
            title='Steps per Episode',
            xaxis_title='Episode',
            yaxis_title='Steps'
        )
        fig_steps.show()

    def run_simulation(self):
        """Complete simulation workflow."""
        self.train(episodes=1000)
        self.visualize_performance_plotly()


# Main execution
if __name__ == "__main__":
    taxi_sim = AdvancedTaxiEnvironment()
    taxi_sim.run_simulation()