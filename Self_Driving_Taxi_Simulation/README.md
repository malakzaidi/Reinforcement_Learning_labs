# 🚖 Taxi-v3 Reinforcement Learning Project 🚕

![image_alt](https://github.com/malakzaidi/web-technologies-tps/blob/main/screenshots/Screenshot%202025-02-26%20021620.png)

## 📖 Overview

This project implements a **Reinforcement Learning (RL)** agent to solve the **Taxi-v3** environment from Gymnasium, a popular library for RL experimentation. The goal is to train an agent to navigate a 5x5 grid, pick up a passenger from one of four locations (Red, Green, Yellow, Blue), and drop them off at a designated destination, all while avoiding obstacles and optimizing for maximum reward.

The project includes multiple RL algorithms (Q-Learning, SARSA, and DQN), visualizations of the learned policy using Plotly, and comparisons across different environments. The focus is on creating an intuitive and interactive visualization to understand the agent's behavior.

---

## 🎯 Objectives

- **Train an RL Agent**: Use Q-Learning to train an agent to solve the Taxi-v3 environment. 🧠
- **Visualize the Policy**: Create an interactive visualization using Plotly to show the agent's optimal actions on the grid. 📊
- **Compare Algorithms**: Compare Q-Learning, SARSA, and DQN in terms of performance (reward and episode length). 📈
- **Experiment with Environments**: Test the agent's performance across different Gymnasium environments (Taxi-v3, FrozenLake-v1, CartPole-v1). 🌍
- **Make It Understandable**: Ensure the visualizations are intuitive, with clear paths, arrows, and markers for actions like pickup and dropoff. 🎨

---

## 🛠️ Setup

### Prerequisites
Make sure you have the following installed:

- **Python 3.8+** 🐍
- **pip** (Python package manager)

### Installation
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Install Dependencies**:
   Create a virtual environment and install the required packages:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   The `requirements.txt` should include:
   ```
   gymnasium==0.29.1
   numpy
   tqdm
   matplotlib
   seaborn
   plotly
   torch
   ```

3. **Verify Installation**:
   Run a simple test to ensure Gymnasium is installed:
   ```bash
   python -c "import gymnasium; print(gymnasium.__version__)"
   ```

---

## 📂 Code Structure

The project consists of a single Python script, `Taxi_RL.py`, with the following structure:

- **Imports** 📦: Libraries for RL (Gymnasium), numerical operations (NumPy), progress tracking (tqdm), visualization (Matplotlib, Seaborn, Plotly), and deep learning (PyTorch).
- **Class `RLAgent`** 🤖:
  - `__init__`: Initializes the environment, Q-table, and learning parameters.
  - `train_q_learning`: Implements the Q-Learning algorithm.
  - `train_sarsa`: Implements the SARSA algorithm.
  - `initialize_dqn` and `train_dqn`: Implements Deep Q-Network (DQN) training.
  - `visualize_taxi_policy`: Visualizes the learned policy for Taxi-v3 using Plotly.
  - `run_visualization`: Runs episodes with the trained agent and renders the environment.
  - `compare_algorithms`: Compares Q-Learning, SARSA, and DQN performance.
  - `experiment_with_environments`: Tests the agent on multiple environments.
- **Main Execution** 🚀: Runs the training, visualization, and comparison steps.

---

## 🚀 Usage

### Running the Project
To train the agent, visualize the policy, and compare algorithms, run the main script:

```bash
python Taxi_RL.py
```

### What Happens When You Run the Script?
1. **Training**:
   - The agent trains using Q-Learning for 5000 episodes.
   - Training progress is shown with a progress bar (via `tqdm`).
   - The average reward over the last 100 episodes is printed.

2. **Policy Visualization**:
   - An interactive Plotly visualization is generated for the "Passenger at Blue, Destination: Red" scenario.
   - The visualization shows:
     - A 5x5 grid with arrows (movement actions), "P" (Pickup), and "D" (Dropoff).
     - Red rectangle: Passenger location (Blue at 4,3).
     - Green rectangle: Destination (Red at 0,0).
     - Purple dashed line: Optimal path from a starting position (2,2).

3. **Run Visualization**:
   - The trained agent runs 3 episodes in the rendered environment, showing its actions in real-time.

4. **Algorithm Comparison**:
   - Compares Q-Learning, SARSA, and DQN over 500 episodes.
   - Plots reward and episode length for each algorithm.

5. **Environment Experiment**:
   - Tests the agent on Taxi-v3, FrozenLake-v1, and CartPole-v1.
   - Plots average rewards for each environment.

---

## 📊 Visualization Details

The policy visualization for Taxi-v3 is designed to be intuitive:

- **Grid**: A 5x5 grid representing the Taxi-v3 environment.
- **Actions**:
  - **Blue Arrows**: Indicate movement (South ↓, North ↑, East →, West ←).
  - **Orange "P"**: Pickup action at the passenger's location.
  - **Green "D"**: Dropoff action at the destination.
- **Markers**:
  - **Red Rectangle**: Passenger's starting location (e.g., Blue at 4,3).
  - **Green Rectangle**: Destination (e.g., Red at 0,0).
- **Path**:
  - **Purple Dashed Line**: The optimal path the taxi takes from a starting position (e.g., 2,2) to pick up the passenger and drop them off.

### Example Visualization
For "Passenger at Blue, Destination: Red":
- The taxi starts at (2,2).
- Moves to (4,3) to pick up the passenger (Blue).
- Moves to (0,0) to drop off the passenger (Red).
- The path avoids walls and follows a logical sequence (e.g., East → South → Pickup → North → West → Dropoff).

---

## 📈 Results Interpretation

### Training Results
- **Average Reward**: The average reward over the last 100 episodes indicates training success.
  - Optimal reward for Taxi-v3: ~8 to 10.
  - If the reward is lower (e.g., 6.95), consider increasing `episodes` or tuning hyperparameters (`alpha`, `epsilon_decay`).

### Visualization
- Check if the purple path makes sense:
  - Does the taxi move toward the passenger (Blue) for pickup?
  - Does it then move toward the destination (Red) for dropoff?
  - Are there erratic movements (e.g., West then up then down)? If so, the policy may need more training.

### Algorithm Comparison
- **Reward Plot**: Higher rewards indicate better performance.
- **Episode Length Plot**: Shorter episodes suggest faster task completion.

### Environment Experiment
- **Bar Chart**: Compare average rewards across environments.
  - Taxi-v3 and FrozenLake-v1 (discrete) should perform better with Q-Learning.
  - CartPole-v1 (continuous) may require DQN for better performance.

---

## 🐞 Troubleshooting

- **Low Average Reward**:
  - Increase `episodes` in `train_q_learning` (e.g., to 7000 or 10000).
  - Adjust `alpha` (e.g., to 0.05) or `epsilon_decay` (e.g., to 0.995).
- **Erratic Path**:
  - Add debugging to `visualize_taxi_policy`:
    ```python
    print("Path:", path)
    print("Policy Grid:\n", policy_grid)
    ```
  - Check if the policy grid (`policy_grid`) shows logical actions (0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Dropoff).
- **Visualization Issues**:
  - Ensure Plotly is installed (`pip install plotly`).
  - Check if the browser opens with the interactive plot.

---

## 🔮 Future Improvements

- **More Algorithms**: Add PPO or A2C for comparison. 🧠
- **Hyperparameter Tuning**: Use grid search to optimize `alpha`, `gamma`, and `epsilon_decay`. ⚙️
- **Enhanced Visualization**: Add hover information for Q-values in the Plotly plot. 📊
- **Path Animation**: Animate the taxi's movement along the path for better understanding. 🎬
- **Support for More Environments**: Extend the project to handle more complex Gymnasium environments. 🌍

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. If you have suggestions or issues, open an issue on the repository.

---

## 📧 Contact

For questions or feedback, reach out via email: [malakzaidi815@gmail.com].

---

Enjoy training your Taxi-v3 agent and exploring the world of reinforcement learning! 🚖💨

---

### How to Use This README
1. Save this as `README.md` in your project directory.
2. If you’re hosting the project on GitHub, it will automatically render with icons and formatting.
3. Replace placeholder text (e.g., `<your-repo-url>`, `your-email@example.com`) with your actual details.
4. Add a `requirements.txt` file with the listed dependencies.
5. Optionally, add a `LICENSE` file with the MIT License text if you choose that license.
