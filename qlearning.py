import numpy as np
import random

class QLearning2DOF:
    """Q-learning tabellare per un braccio 2-DOF in C-space discreto"""

    def __init__(self, cspace, start=(0, 0), goal=None, num_actions=4,
                 alpha=0.1, gamma=0.9, epsilon=0.9):
        self.cspace = cspace
        self.N1 = cspace.N1
        self.N2 = cspace.N2
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.start = start
        self.goal = goal if goal else (self.N1-1, self.N2-1)

        # Controlla che lo start sia libero
        if self.cspace.grid[self.start[0], self.start[1]] == 1:
            raise ValueError("Start position is inside an obstacle!")

        # Q-table inizializzata leggermente positiva
        self.Q = np.ones((self.N1, self.N2, num_actions)) * 0.01

    def step(self, state, action):
        i, j = state
        if action == 0:    # theta1+
            i_new = min(i+1, self.N1-1)
            j_new = j
        elif action == 1:  # theta1-
            i_new = max(i-1, 0)
            j_new = j
        elif action == 2:  # theta2+
            i_new = i
            j_new = min(j+1, self.N2-1)
        elif action == 3:  # theta2-
            i_new = i
            j_new = max(j-1, 0)

        # Collisione con ostacolo
        if self.cspace.grid[i_new, j_new] == 1:
            reward = -100
            done = True
        # Arrivo al goal
        elif (i_new, j_new) == self.goal:
            reward = 100
            done = True
        else:
            reward = -0.1  # piccolo negativo per passo libero
            done = False

        return (i_new, j_new), reward, done

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions-1)
        i, j = state
        return np.argmax(self.Q[i, j, :])

    def train(self, num_episodes=2000, max_steps=500, verbose=True):
        for episode in range(num_episodes):
            state = self.start
            for step_count in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done = self.step(state, action)

                i, j = state
                i_next, j_next = next_state

                self.Q[i, j, action] += self.alpha * (
                    reward + self.gamma * np.max(self.Q[i_next, j_next, :]) - self.Q[i, j, action]
                )

                state = next_state
                if done:
                    break

            # Stampa dei primi episodi e ogni 50 episodi
            if verbose and (episode < 5 or episode % 50 == 0):
                print(f"Episode {episode}: finished in {step_count+1} steps, last state {state}")

            # Decay epsilon per ridurre esplorazione progressivamente
            self.epsilon = max(0.05, self.epsilon * 0.995)

    def get_path(self, start=None, max_steps=500):
        state = start if start else self.start
        path = [state]
        for _ in range(max_steps):
            i, j = state
            action = np.argmax(self.Q[i, j, :])
            next_state, reward, done = self.step(state, action)
            path.append(next_state)
            state = next_state
            if done:
                break
        return path
