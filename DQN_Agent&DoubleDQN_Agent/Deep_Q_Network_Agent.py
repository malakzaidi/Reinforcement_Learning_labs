import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

GRID_SIZE = 4
STATE_SIZE = GRID_SIZE * GRID_SIZE
ACTION_SIZE = 4
GAMMA = 0.9
LEARNING_RATE = 0.01
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 2000
EPISODES = 1000

MOVES = {
0: (-1, 0),
1: (1, 0),
2: (0, -1),
3: (0, 1)
}

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
