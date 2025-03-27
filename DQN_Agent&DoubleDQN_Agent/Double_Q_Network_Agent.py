import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

# Paramètres de l'environnement et de l'apprentissage
GRID_SIZE = 4
STATE_SIZE = GRID_SIZE * GRID_SIZE
ACTION_SIZE = 4  # Haut, Bas, Gauche, Droite
GAMMA = 0.9  # Facteur de réduction
LEARNING_RATE = 0.01
EPSILON = 1.0  # Taux d'exploration initial
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 2000
EPISODES = 1000

# Définition des mouvements possibles
MOVES = {
    0: (-1, 0),  # Haut
    1: (1, 0),   # Bas
    2: (0, -1),  # Gauche
    3: (0, 1)    # Droite
}

# Environnement GridWorld
class GridWorld:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.reset()

    def reset(self):
        """Réinitialise l'agent à la position de départ."""
        self.agent_pos = (0, 0)
        self.goal_pos = (3, 3)
        self.obstacle_pos = (1, 1)  # Une case d'obstacle
        return self.get_state()

    def get_state(self):
        """Retourne l'état sous forme d'un vecteur binaire."""
        state = np.zeros((self.grid_size, self.grid_size))
        state[self.agent_pos] = 1
        return state.flatten()

    def step(self, action):
        """Fait avancer l'agent et renvoie (nouvel état, récompense, terminé)."""
        x, y = self.agent_pos
        dx, dy = MOVES[action]
        new_x, new_y = x + dx, y + dy

        # Vérifier les limites
        if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
            self.agent_pos = (new_x, new_y)

        # Vérifier la récompense
        if self.agent_pos == self.goal_pos:
            return self.get_state(), 10, True  # Objectif atteint
        elif self.agent_pos == self.obstacle_pos:
            return self.get_state(), -5, False  # Obstacle
        else:
            return self.get_state(), -1, False  # Déplacement normal

# Agent Double DQN
class DoubleDQNAgent:
    def __init__(self):
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Construit le réseau de neurones."""
        model = Sequential([
            Dense(24, activation='relu', input_shape=(self.state_size,)),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    def update_target_model(self):
        """Copie les poids du modèle principal vers le modèle cible."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Stocke une expérience dans la mémoire."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choisit une action en suivant une stratégie ε-greedy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Exploration
        q_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])  # Exploitation

    def replay(self):
        """Entraîne le modèle avec des expériences passées."""
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            target = self.model.predict(np.array([state]), verbose=0)[0]
            if done:
                target[action] = reward
            else:
                # Sélection de l'action avec le réseau principal (online)
                next_action = np.argmax(self.model.predict(np.array([next_state]), verbose=0)[0])
                # Évaluation de l'action avec le réseau cible (target)
                target[action] = reward + GAMMA * self.target_model.predict(np.array([next_state]), verbose=0)[0][next_action]
            self.model.fit(np.array([state]), np.array([target]), epochs=1, verbose=0)
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY  # Réduction de l'exploration
        self.update_target_model()

# Entraînement de l'agent
env = GridWorld()
agent = DoubleDQNAgent()

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    for step in range(50):  # Limite de 50 déplacements
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
    agent.replay()
    print(f"Épisode {episode + 1}/{EPISODES}, Score: {total_reward}, Epsilon: {agent.epsilon:.4f}")

# Sauvegarde du modèle
agent.model.save("double_dqn_model.keras")

# Chargement du modèle entraîné
agent.model.load_weights("double_dqn_model.keras")

# Test de l'agent
env = GridWorld()
state = env.reset()
total_reward = 0
for step in range(50):  # Limite de 50 déplacements
    action = agent.act(state)
    next_state, reward, done = env.step(action)
    state = next_state
    total_reward += reward
    if done:
        break
print(f"Score de Test: {total_reward}")