import numpy as np
import random

class QLearning2DOF:
    """Tabular Q-learning for 2DOF planar arm in C-space"""

    def __init__(self, cspace, start=(0, 0), goal=None, num_actions=4,
                 alpha=0.1, gamma=0.95, epsilon=0.9):
        self.cspace = cspace
        self.N1 = cspace.N1
        self.N2 = cspace.N2
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_initial = epsilon
        self.epsilon = epsilon

        self.start = start
        self.goal = goal if goal else (self.N1-1, self.N2-1)

        # Controll if start or goal are in obstacles
        if self.cspace.grid[self.start[0], self.start[1]] == 1:
            raise ValueError("Start position is inside an obstacle!")
        
        if self.cspace.grid[self.goal[0], self.goal[1]] == 1:
            raise ValueError("Goal position is inside an obstacle!")

        
        self.Q = np.zeros((self.N1, self.N2, num_actions))
        
        # Count visits to states for analysis
        self.visit_count = np.zeros((self.N1, self.N2))

    def step(self, state, action):
        """Executes action in the environment, returns next_state, reward, done"""
        i, j = state
        
        # Calcola nuovo stato
        if action == 0:    # theta1+
            i_new = min(i+1, self.N1-1)
            j_new = j
        elif action == 1:  # theta1-
            i_new = max(i-1, 0)
            j_new = i
        elif action == 2:  # theta2+
            i_new = i
            j_new = min(j+1, self.N2-1)
        elif action == 3:  # theta2-
            i_new = i
            j_new = max(j-1, 0)
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check collision
        if self.cspace.grid[i_new, j_new] == 1:
            # Return to the previous state with a penalty, NOT done
            return state, -10.0, False
        
        # Check if goal reached
        if (i_new, j_new) == self.goal:
            return (i_new, j_new), 100.0, True
        
        # Movements: small penalty to encourage shorter paths
        # Uses Manhattan distance to goal as euristic
        dist_to_goal = abs(i_new - self.goal[0]) + abs(j_new - self.goal[1]) # check Manhattan
        reward = -1.0 - 0.01 * dist_to_goal
        
        return (i_new, j_new), reward, False

    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            # Exploration: casual action
            return random.randint(0, self.num_actions-1)
        else:
            # Exploitation: best action
            i, j = state
            q_values = self.Q[i, j, :]
            
            max_q = np.max(q_values)
            
            # Tie-breaking: if there are multiple best actions, choose randomly among them
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)

    def train(self, num_episodes=2000, max_steps=1000, verbose=True):
        """Training loop with epsilon decay"""
        success_count = 0
        
        for episode in range(num_episodes):
            state = self.start
            total_reward = 0
            
            for step_count in range(max_steps):
                # Visit count
                self.visit_count[state[0], state[1]] += 1
                
                # Choose action
                action = self.choose_action(state)
                
                # Execute action
                next_state, reward, done = self.step(state, action)
                total_reward += reward
                
                # Q-learning update
                i, j = state
                i_next, j_next = next_state
                
                best_next_q = np.max(self.Q[i_next, j_next, :])
                td_target = reward + self.gamma * best_next_q
                td_error = td_target - self.Q[i, j, action]
                self.Q[i, j, action] += self.alpha * td_error
                
                # Next state
                state = next_state
                
                if done:
                    if reward > 0:  # Goal reached
                        success_count += 1
                    break
            
            # Verbose output
            if verbose and (episode < 10 or episode % 100 == 0):
                success_rate = 100.0 * success_count / (episode + 1)
                print(f"Episode {episode}: steps={step_count+1}, reward={total_reward:.2f}, "
                      f"epsilon={self.epsilon:.3f}, success_rate={success_rate:.1f}%")
            
            # Slower epsilon decay 
            self.epsilon = max(0.05, self.epsilon * 0.9995)
        
        print(f"\nTraining completed. Final success rate: {100.0*success_count/num_episodes:.1f}%")
        print(f"Visited states: {np.sum(self.visit_count > 0)} / {self.N1 * self.N2}")

    def get_path(self, start=None, max_steps=1000):
        """Extract the greedy path from the Q-table"""
        state = start if start else self.start
        path = [state]
        visited = {state}  
        
        for step in range(max_steps):
            i, j = state
            
            # Choose best action
            action = np.argmax(self.Q[i, j, :])
            
            # Execute action
            next_state, reward, done = self.step(state, action)
            
            # Check for loops
            if next_state in visited:
                print(f"WARNING: Loop detected at step {step}, state {next_state}")
                print(f"State {state} Q-value: {self.Q[i, j, :]}")
                break
            
            path.append(next_state)
            visited.add(next_state)
            state = next_state
            
            if done:
                if reward > 0:
                    print(f"Goal reached in {len(path)} steps!")
                break
        
        if state != self.goal:
            print(f"WARNING: Goal NOT reached. Last state: {state}, Goal: {self.goal}")
        
        return path