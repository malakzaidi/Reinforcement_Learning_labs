# 🚖 Self driving Taxi-v3 Reinforcement Learning Project with Open AI Gymnasium 🚕

Welcome to the **Taxi-v3 Reinforcement Learning Project**, a sophisticated and extensible framework designed to tackle the classic Taxi-v3 environment using **Open AI Gymnasium**, the modern evolution of the Open AI Gym library for reinforcement learning (RL) experimentation. This project provides researchers, developers, and enthusiasts with a comprehensive toolkit for training, visualizing, and comparing RL agents, emphasizing actionable insights through interactive visualizations and performance analysis across multiple environments. 

![image_alt](https://github.com/malakzaidi/Reinforcement_Learning_projects/blob/main/Self_Driving_Taxi_Simulation/visualizations/Taxi%202025-03-08%2017-14-12.gif)

## Table of Contents
- [📋 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🧩 Project Architecture](#-project-architecture)
- [🛠️ Installation](#️-installation)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
- [🚀 Getting Started](#-getting-started)
  - [Basic Usage](#basic-usage)
  - [Algorithm Comparison](#algorithm-comparison)
  - [Customizing Hyperparameters](#customizing-hyperparameters)
- [📖 About Open AI Gymnasium](#-about-open-ai-gymnasium)
- [🔬 Algorithm Deep Dive](#-algorithm-deep-dive)
  - [Q-Learning](#q-learning)
  - [SARSA](#sarsa)
  - [DQN (Deep Q-Network)](#dqn-deep-q-network)
- [📊 Visualization Capabilities](#-visualization-capabilities)
  - [Policy Visualization](#policy-visualization)
  - [Performance Analytics](#performance-analytics)
- [🔍 Example Results & Analysis](#-example-results--analysis)
  - [Taxi-v3 Environment](#taxi-v3-environment)
  - [Algorithm Comparison (500 Episodes)](#algorithm-comparison-500-episodes)
  - [Learning Dynamics](#learning-dynamics)
- [🧪 Experimental Results](#-experimental-results)
- [📚 API Reference](#-api-reference)
  - [RLAgent Class](#rlagent-class)
- [🔧 Deployment](#-deployment)
  - [Local Development](#local-development)
  - [Containerization (Future)](#containerization-future)
- [🔮 Future Development](#-future-development)
- [📚 References](#-references)
- [🙏 Acknowledgments](#-acknowledgments)
- [🐛 Troubleshooting](#-troubleshooting)
- [🤝 Contributing Guidelines](#-contributing-guidelines)
- [📧 Contact](#-contact)



<h2 align="center">🚀 Technologies Used in This Project</h2>

<p align="center">
  <!-- Python -->
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="50" alt="Python"/>

  <!-- PyTorch -->
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" height="50" alt="PyTorch"/>
  
  <!-- NumPy -->
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" height="50" alt="NumPy"/>
  
  <!-- Matplotlib -->
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/84/Matplotlib_icon.svg" height="50" alt="Matplotlib"/>
  
  <!-- Seaborn -->
  <img src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" height="50" alt="Seaborn"/>

  <!-- tqdm -->
  <img src="https://img.icons8.com/fluency/48/progress-indicator.png" height="50" alt="tqdm"/>
  
  <!-- Build Status -->
  <img src="https://img.icons8.com/color/48/000000/checkmark.png" height="50" alt="Build Status"/>
</p>



---

## 📋 Overview

The **Taxi-v3 Reinforcement Learning Project** leverages **Open AI Gymnasium** to implement and evaluate RL algorithms on the Taxi-v3 environment. In Taxi-v3, an agent (a taxi) navigates a 5x5 grid to pick up a passenger from one of four locations (Red, Green, Yellow, Blue) and deliver them to a designated destination, avoiding obstacles and optimizing for maximum cumulative reward. The framework implements three foundational RL algorithms—Q-Learning, SARSA, and Deep Q-Networks (DQN)—and offers advanced visualization capabilities using Plotly to interpret the learned policies. It also supports experimentation across diverse Gymnasium environments, making it a versatile platform for RL research and education.

---

## ✨ Key Features

| 🔍 Feature                  | 📝 Description                                                                 |
|-----------------------------|--------------------------------------------------------------------------------|
| **Multi-Algorithm Support** | Implements Q-Learning, SARSA, and DQN with configurable settings.              |
| **Gymnasium Integration**   | Seamlessly integrates with Open AI Gymnasium for environment management.       |
| **Interactive Visualization** | Rich, interactive policy maps and performance analytics using Plotly.          |
| **Comparative Analysis**    | Built-in benchmarking tools to evaluate algorithm performance.                 |
| **Live Demonstrations**     | Real-time rendering of agent behavior in Gymnasium environments.               |
| **Cross-Environment Testing** | Supports experimentation across Taxi-v3, FrozenLake-v1, and CartPole-v1.       |
| **Extensible Architecture** | Modular design for integrating new algorithms or custom environments.          |

---

## 🧩 Project Architecture

The peoject is built around the `RLAgent` class, following a modular, object-oriented design:

```
RLAgent
├── 📦 Environment Handling (Open AI Gymnasium)
│   ├── State Space Management (Discrete/Continuous)
│   └── Action Space Adaptation
├── ⚙️ Algorithm Implementation
│   ├── Q-Learning (Tabular)
│   ├── SARSA (On-Policy TD)
│   └── DQN (Deep Learning)
│       ├── Neural Network (2-Layer MLP)
│       └── Experience Replay Buffer
├── ⏳ Training Pipeline (Configurable Episodes/Batches)
├── 🎨 Visualization Tools (Plotly-Based)
└── 🔬 Experiment Management (Benchmarking & Analysis)
```

---

## 🛠️ Installation

### Prerequisites
- **Python**: Version 3.8 or higher.
- **pip**: Python package manager.

### Installation Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/taxi-rl-project.git
   cd taxi-rl-project
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Install the required packages, including Open AI Gymnasium, using the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

   **requirements.txt**:
   ```
   gymnasium>=0.26.3
   numpy>=1.20.0
   matplotlib>=3.4.0
   seaborn>=0.11.0
   plotly>=5.3.0
   torch>=1.10.0
   tqdm>=4.62.0
   ```

4. **Verify Installation**:
   Confirm that Open AI Gymnasium and other dependencies are installed:
   ```bash
   python -c "import gymnasium, torch, plotly; print('Dependencies verified successfully!')"
   ```

---

## 🚀 Getting Started

### Basic Usage
```python
from rl_agent import RLAgent

# Initialize and train a Q-Learning agent on Taxi-v3
agent = RLAgent(env_name='Taxi-v3')
agent.train_q_learning(episodes=5000)

# Visualize the learned policy
agent.visualize_taxi_policy()

# Run live demonstration
agent.run_visualization(episodes=3)
```

### Algorithm Comparison
```python
# Compare Q-Learning, SARSA, and DQN
results = agent.compare_algorithms(
    env_name='Taxi-v3',
    algorithms=['q_learning', 'sarsa', 'dqn'],
    episodes=500
)
```

### Customizing Hyperparameters
```python
agent = RLAgent(env_name='Taxi-v3')
agent.learning_params = {
    'alpha': 0.2,           # Learning rate
    'gamma': 0.95,          # Discount factor
    'epsilon': 1.0,         # Initial exploration rate
    'epsilon_decay': 0.995, # Exploration decay rate
    'min_epsilon': 0.01,    # Minimum exploration rate
}
agent.train_q_learning(episodes=3000)
```

---

## 📖 About Open AI Gymnasium

**Open AI Gymnasium** is a modern, actively maintained fork of the original Open AI Gym library, designed for RL experimentation. It provides a standardized interface for interacting with environments like Taxi-v3, FrozenLake-v1, and CartPole-v1, offering features such as:
- **Consistent API**: For defining state spaces, action spaces, and reward structures.
- **Rendering Support**: For visualizing agent behavior in real-time.
- **Extensive Documentation**: Available at [gymnasium.farama.org](https://gymnasium.farama.org/).
- **Community Support**: Regular updates and contributions from the RL community.

This project uses Gymnasium to ensure compatibility with the latest RL standards and to leverage its enhanced features for environment management and visualization.

---

## 🔬 Algorithm Deep Dive

### Q-Learning
Q-Learning is a model-free, off-policy RL algorithm that learns the action-value function:

\[ Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)] \]

- **Features**:
  - 🎯 Epsilon-greedy exploration with adaptive decay.
  - 📈 Dynamic learning rate adjustment.
  - 📊 Real-time Q-value tracking.
  - 🔄 Optimized for large discrete state spaces in Gymnasium environments.

### SARSA
SARSA (State-Action-Reward-State-Action) is an on-policy TD learning algorithm:

\[ Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)] \]

- **Highlights**:
  - 🧮 On-policy updates reflecting actual policy behavior.
  - ⚖️ Balanced exploration-exploitation tradeoff.
  - 🛡️ Conservative policy suitable for stochastic environments.
  - 📊 Comprehensive performance metrics.

### DQN (Deep Q-Network)
DQN extends Q-learning with a neural network to approximate the Q-function:

```python
self.dqn_model = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, self.action_space)
)
```

- **Advanced Features**:
  - 🧠 Two-layer MLP architecture.
  - 🔄 Experience replay buffer to decorrelate samples.
  - 🎯 Target network for training stability.
  - 📊 Batch learning with optimized tensor operations.

---

## 📊 Visualization Capabilities

### Policy Visualization
For Taxi-v3, the framework delivers an interactive policy map using Plotly, featuring:
- 🧭 **Directional Arrows**: Indicate optimal movements (South ↓, North ↑, East →, West ←).
- 🚕 **Path Traces**: Purple dashed lines show the agent's optimal trajectory.
- 🏁 **Action Markers**: "P" for pickup and "D" for dropoff locations.
- 🎯 **Location Indicators**: Red rectangles for passenger start (e.g., Blue at [4,3]), green for destination (e.g., Red at [0,0]).

**Example Visualization: Policy for Passenger at Blue, Destination: Red**

![Policy Visualization](images/policy_visualization.png)

In this visualization:
- The taxi starts at position [2,2].
- The path (purple dashed line) moves from [2,2] to [3,3] → [3,4] → [2,4], where it picks up the passenger (marked "P" in orange) at [4,3] (Blue).
- The taxi then moves toward the destination at [0,0] (Red, marked with a green rectangle and "D" for dropoff).
- Blue arrows indicate the learned policy's movement directions at each grid position.
- The path does not directly reach the passenger or destination, suggesting the policy needs further training to optimize the route.

### Performance Analytics
- 📈 **Rewards per Episode**: Tracks learning progress.
- ⏱️ **Episode Completion Time**: Measures efficiency.
- 📊 **Comparative Metrics**: Plots algorithm performance.
- 🔄 **Convergence Analysis**: Assesses training stability.

![image_alt](https://github.com/malakzaidi/Reinforcement_Learning_projects/blob/main/Self_Driving_Taxi_Simulation/visualizations/Screenshot%202025-03-08%20204137.png)

**Algorithm Comparison: Rewards and Episode Length**

![Algorithm Comparison](https://github.com/malakzaidi/Reinforcement_Learning_projects/blob/main/Self_Driving_Taxi_Simulation/visualizations/Screenshot%202025-03-08%20180752.png)

This plot compares Q-Learning, SARSA, and DQN over 500 episodes:
- **Left (Rewards)**: All algorithms show initial negative rewards due to exploration, with Q-Learning (blue) and SARSA (green) converging to around -66 to -68, while DQN (orange) achieves a higher average reward of -5.50.
- **Right (Episode Length)**: Episode lengths decrease over time, indicating improved efficiency, though fluctuations suggest variability in learning stability.
- **Analysis**: DQN outperforms tabular methods in terms of reward, but Q-Learning and SARSA converge faster in this setting.

**Training Progress: Q-Learning Rewards**

![Training Progress](https://github.com/malakzaidi/Reinforcement_Learning_projects/blob/main/Self_Driving_Taxi_Simulation/visualizations/Screenshot%202025-03-08%20121901.png)

This plot shows the total reward per episode during Q-Learning training:
- The reward starts highly negative (around -800) due to exploration and incorrect actions.
- It improves steadily, reaching around -200 by episode 500, with significant variability.
- **Analysis**: The agent is learning, but the reward has not yet reached the optimal range (8–10), indicating that more episodes or hyperparameter tuning may be needed.

**Performance Across Different Environments**

![Performance Across Environments](https://github.com/malakzaidi/Reinforcement_Learning_projects/blob/main/Self_Driving_Taxi_Simulation/visualizations/Screenshot%202025-03-08%20191759.png)

This bar chart compares average rewards across environments:
- **Taxi-v3**: Achieves an average reward of ~7.28, indicating decent but suboptimal performance.
- **FrozenLake-v1**: Scores 0.00, reflecting the sparse reward structure and difficulty of the environment.
- **CartPole-v1**: Achieves ~153.83, showing better performance due to DQN's suitability for continuous state spaces.
- **Analysis**: The framework performs best on CartPole-v1, but Taxi-v3 needs further optimization, and FrozenLake-v1 requires a different approach (e.g., reward shaping).

---

## 🔍 Example Results & Analysis

### Taxi-v3 Environment
- ✅ **Average Reward**: 7.28–7.46 after 5000 episodes.
- ⏱️ **Average Steps**: 10–14 steps per episode.
- 🔄 **Convergence**: Steady improvement, with refinement after 200 episodes.

The policy visualization (above) shows a path that does not directly reach the passenger or destination, indicating the need for more training or hyperparameter tuning.

### Algorithm Comparison (500 Episodes)
| Algorithm  | Average Reward (Last 100) | Strengths                          |
|------------|---------------------------|------------------------------------|
| **Q-Learning** 🥇 | -66.13                   | Fast convergence, robust for Taxi-v3 |
| **SARSA** 🥈       | -68.95                   | Stable, conservative policies       |
| **DQN** 🥉         | -5.50                    | Potential for complex tasks         |

### Learning Dynamics
- **Initial Phase (0–200 episodes)**: Steep reward increase.
- **Intermediate Plateau (200–300 episodes)**: Policy refinement.
- **Convergence (300–500 episodes)**: Diminishing returns.

---

## 🧪 Experimental Results

| Environment  | Average Reward (Last 100 Episodes) | Best Algorithm |
|--------------|-----------------------------------|----------------|
| Taxi-v3      | 7.28                               | Q-Learning     |
| FrozenLake-v1| 0.00                               | N/A            |
| CartPole-v1  | 153.83                             | DQN            |

- **Insights**: Q-Learning excels in discrete tasks like Taxi-v3, while DQN performs better in continuous settings like CartPole-v1. FrozenLake-v1's sparse rewards pose a challenge.

---

## 📚 API Reference

### `RLAgent` Class
- **`__init__(env_name='Taxi-v3')`**:
  - Initializes the agent with a Gymnasium environment.
  - **Args**: `env_name` (str) – Environment ID (default: 'Taxi-v3').
- **`train_q_learning(episodes=5000)`**:
  - Trains with Q-Learning.
  - **Returns**: Float – Average reward over last 100 episodes.
- **`visualize_taxi_policy()`**:
  - Generates an interactive policy visualization for Taxi-v3.
- **`run_visualization(episodes=5, max_steps=100)`**:
  - Renders agent performance in real-time using Gymnasium rendering.
- **`compare_algorithms(algorithms=['q_learning', 'sarsa', 'dqn'], episodes=500)`**:
  - Benchmarks specified algorithms.
- **`experiment_with_environments(env_names=['Taxi-v3', 'FrozenLake-v1', 'CartPole-v1'])`**:
  - Evaluates performance across Gymnasium environments.

---

## 🔧 Deployment

### Local Development
- Run directly with `python Taxi_RL.py`.
- Ensure a compatible GPU for DQN (optional).

### Containerization (Future)
- Planned Docker support for reproducible deployments:
  ```dockerfile
  FROM python:3.8-slim
  COPY . /app
  WORKDIR /app
  RUN pip install -r requirements.txt
  CMD ["python", "Taxi_RL.py"]
  ```

---

## 🔮 Future Development

- 🧠 **Algorithm Expansion**: Add PPO, SAC, and A2C.
- 🌐 **Multi-Agent Support**: Enable cooperative/competitive scenarios.
- 🎛️ **Hyperparameter Optimization**: Integrate Bayesian optimization.
- 📊 **Advanced Visualization**: 3D plots and dashboards.
- 🧩 **Custom Environment Builder**: Module for custom Gymnasium environments.

---

## 📚 References
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. *Nature*, 518(7540), 529–533.
- Gymnasium Documentation: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)

---

## 🙏 Acknowledgments
- **Open AI Gymnasium Team**: For maintaining a robust RL environment library.
- **DeepMind & OpenAI**: For pioneering RL research.
- **Community Contributors**: For ongoing feedback and support.

---

## 🐛 Troubleshooting

- **Low Reward**: Increase `episodes` to 7000–10000 or tune `alpha` (e.g., 0.05) and `epsilon_decay` (e.g., 0.995).
- **Visualization Issues**: Ensure Plotly is installed and a browser is available.
- **Debugging**: Use `print("Path:", path)` in `visualize_taxi_policy` to inspect the path.

---

## 🤝 Contributing Guidelines

1. **Fork the Repository**.
2. **Create a Feature Branch**: `git checkout -b feature/<feature-name>`.
3. **Commit Changes**: `git commit -m "Add <feature>"`.
4. **Push and Submit**: `git push origin feature/<feature-name>` and open a PR.
5. **Follow Code Style**: Adhere to PEP 8.

For issues, use the [issue tracker](https://github.com/yourusername/taxi-rl-project/issues).

---

## 📧 Contact
- **Email**: [malakzaidi815@gmail.com](malakzaidi815@gmail.com)
- **GitHub**: [@malakzaidi](https://github.com/malakzaidi)

---
