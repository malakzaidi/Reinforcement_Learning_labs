import time
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from itertools import product
import torch
import torch.nn as nn
import torch.optim as optim

class RLAgent:
    def __init__(self, env_name='Taxi-v3'):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.render_env = gym.make(env_name, render_mode="human")

        self.action_space = self.env.action_space.n

        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            self.state_space = self.env.observation_space.n
            self.discrete_state = True
            self.Q = np.zeros((self.state_space, self.action_space))
        else:
            self.state_space = self.env.observation_space.shape[0]
            self.discrete_state = False
            self.Q = None

        self.learning_params = {
            'alpha': 0.8,
            'gamma': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'min_epsilon': 0.01,
        }

        self.training_rewards = []
        self.episode_lengths = []

    # [Previous methods like train_q_learning, train_sarsa, train_dqn, etc., remain unchanged]

    def visualize_taxi_policy(self):
        if self.env_name != 'Taxi-v3':
            print("Policy visualization is only available for Taxi-v3 environment.")
            return

        # Grid size and location definitions
        grid_size = 5
        passenger_locations = [(0, 0), (0, 4), (4, 0), (4, 3)]  # R, G, Y, B
        destination_locations = passenger_locations.copy()
        action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]
        action_colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow']  # Unique colors for actions

        # Create figures for different passenger-destination combinations
        passenger_positions = [0, 1, 2, 3]  # R, G, Y, B
        dest_position = 0  # Destination fixed at Red for all examples

        for passenger_pos in passenger_positions:
            # Initialize policy and value grids
            policy_grid = np.zeros((grid_size, grid_size), dtype=int)
            value_grid = np.zeros((grid_size, grid_size))
            q_value_details = np.zeros((grid_size, grid_size, self.action_space))

            # Populate grids
            for row in range(grid_size):
                for col in range(grid_size):
                    state = row * 5 + col + passenger_pos * 25 + dest_position * 125
                    q_values = self.Q[state, :]
                    best_action = np.argmax(q_values)
                    policy_grid[row, col] = best_action
                    value_grid[row, col] = np.max(q_values)
                    q_value_details[row, col, :] = q_values

            # Create interactive Plotly figure
            fig = go.Figure()

            # Add heatmap for Q-values
            fig.add_trace(
                go.Heatmap(
                    z=value_grid,
                    x=list(range(grid_size)),
                    y=list(range(grid_size)),
                    colorscale='Viridis',
                    colorbar_title="Max Q-Value",
                    zmin=np.min(value_grid),
                    zmax=np.max(value_grid),
                    showscale=True
                )
            )

            # Add annotations for best actions
            annotations = []
            for row in range(grid_size):
                for col in range(grid_size):
                    action_idx = policy_grid[row, col]
                    text = action_names[action_idx]
                    color = action_colors[action_idx]
                    annotations.append(
                        dict(
                            x=col,
                            y=grid_size - 1 - row,  # Invert y-axis to match gym coordinates
                            xref="x",
                            yref="y",
                            text=text,
                            showarrow=False,
                            font=dict(color=color, size=12),
                            align="center"
                        )
                    )

            # Add passenger and destination markers
            p_row, p_col = passenger_locations[passenger_pos]
            d_row, d_col = destination_locations[dest_position]
            fig.add_shape(
                type="rect",
                x0=p_col - 0.5, y0=grid_size - 1 - p_row - 0.5,
                x1=p_col + 0.5, y1=grid_size - 1 - p_row + 0.5,
                line=dict(color="red", width=2),
                fillcolor="rgba(255, 0, 0, 0.2)"
            )
            fig.add_shape(
                type="rect",
                x0=d_col - 0.5, y0=grid_size - 1 - d_row - 0.5,
                x1=d_col + 0.5, y1=grid_size - 1 - d_row + 0.5,
                line=dict(color="green", width=2),
                fillcolor="rgba(0, 255, 0, 0.2)"
            )

            # Update layout
            passenger_labels = ['Red', 'Green', 'Yellow', 'Blue']
            destination_labels = ['Red', 'Green', 'Yellow', 'Blue']
            fig.update_layout(
                title=f'Policy for Passenger at {passenger_labels[passenger_pos]}, Destination: {destination_labels[dest_position]}',
                xaxis_title="Column",
                yaxis_title="Row",
                annotations=annotations,
                showlegend=False,
                height=600,
                width=600
            )

            # Add hover information for all Q-values
            fig.update_traces(
                hovertemplate=
                "Row: %{y}<br>Col: %{x}<br>" +
                "<br>".join([f"{action_names[i]} Q-value: %{{customdata[{i}]}}"
                            for i in range(self.action_space)]) + "<br>",
                customdata=q_value_details
            )

            # Show the figure
            fig.show()

    # [Rest of the methods like run_visualization, experiment_with_environments, etc., remain unchanged]

if __name__ == "__main__":
    agent = RLAgent(env_name='Taxi-v3')

    # Train with Q-learning
    agent.train_q_learning(episodes=1000)

    # Visualize the learned policy and show agent in action
    agent.visualize_taxi_policy()
    agent.run_visualization(episodes=3)

    # Compare different algorithms
    agent.compare_algorithms(episodes=500)

    # Experiment with different environments
    agent.experiment_with_environments(['Taxi-v3', 'FrozenLake-v1', 'CartPole-v1'])