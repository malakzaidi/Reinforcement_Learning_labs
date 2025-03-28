# 🤖 Deep Q-Learning Grid World Navigation Agents

## 🌟 Project Overview

This project demonstrates two powerful reinforcement learning approaches for solving a Grid World navigation problem using Deep Q-Networks (DQN) and Double Deep Q-Networks (Double DQN).

### 🧠 Key Features
- Implemented both standard DQN and Double DQN algorithms
- Navigates a 4x4 grid environment
- Uses TensorFlow and Keras for neural network implementation
- Adaptive exploration strategy with epsilon-greedy method
- Handles obstacles and goal-seeking behavior

## 📍 Environment Description

### Grid World Characteristics
- **Grid Size**: 4x4
- **Agent Starting Position**: (0, 0)
- **Goal Position**: (3, 3)
- **Obstacle Position**: (1, 1)

### Action Space
- 4 Possible Actions:
  1. Up
  2. Down
  3. Left
  4. Right

### Reward Structure
- **Goal Reached**: +10 points
- **Obstacle Hit**: -5 points
- **Normal Move**: -1 point

## 🚀 Key Reinforcement Learning Concepts

### DQN (Deep Q-Network)
- Uses a neural network to approximate Q-values
- Learns through experience replay
- Gradually reduces exploration with epsilon decay

### Double DQN
- Introduces a separate target network
- Reduces overestimation of Q-values
- Improves learning stability

## 🔧 Technical Details

### Neural Network Architecture
- Input Layer: State representation (16 neurons)
- Hidden Layers: 
  - First hidden layer: 24 neurons (ReLU activation)
  - Second hidden layer: 24 neurons (ReLU activation)
- Output Layer: Action Q-values (4 neurons, linear activation)

### Hyperparameters
- **Discount Factor (GAMMA)**: 0.9
- **Learning Rate**: 0.01
- **Initial Exploration Rate (EPSILON)**: 1.0
- **Minimum Exploration Rate**: 0.01
- **Exploration Decay Rate**: 0.995
- **Batch Size**: 32
- **Memory Size**: 2000
- **Training Episodes**: 1000

## 🏃‍♂️ Training Process

1. Initialize the environment and agent
2. For each episode:
   - Reset the environment
   - Agent explores/exploits the grid
   - Collect experiences
   - Update neural network weights
   - Decay exploration rate

## 📊 Performance Tracking
- Prints episode number
- Shows total reward per episode
- Tracks exploration rate (epsilon)

## 💾 Model Persistence
- Trained models saved as:
  - `dqn_model.keras`
  - `double_dqn_model.keras`

## 🔍 Key Differences Between Implementations

### Standard DQN
- Single neural network for action selection and evaluation
- More prone to Q-value overestimation

### Double DQN
- Separate online and target networks
- More stable learning
- Reduced overestimation bias

## 🚦 Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow
- NumPy

### Installation
```bash
pip install tensorflow numpy
```

### Running the Agents
```bash
python Deep_Q_Network_Agent.py
python Double_Q_Network_Agent.py
```

## 🌈 Future Improvements
- Implement Prioritized Experience Replay
- Add more complex grid environments
- Experiment with different network architectures
- Incorporate visual state representations


## 🙏 Acknowledgments
Inspired by the groundbreaking work in Deep Reinforcement Learning.

---

**Happy Learning! 🎓🤖**
