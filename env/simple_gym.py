import numpy as np

class TwoDimArrayMap:
    def __init__(self, x_dim, y_dim, action_space_dim = 4):
        self.maze = np.zeros([x_dim, y_dim])
        self.success_reward = 0
        self.failed_reward = -1
        self.reward_states = np.full_like(np.zeros([x_dim, y_dim]), self.failed_reward)
        
        self.state = np.array([0, 0])
        self.observation_space_dim = x_dim * y_dim
        self.action_space_dim = action_space_dim
        self.row = len(self.maze)
        self.col = len(self.maze[0])
    
    def SimpleAntMazation(self):   # simple maze having one large wall. Represent wall as 1. At the reward states, wall is -9
        for i in range(self.row):
            for j in range(self.col):
                if (self.row//3) <= i < (2 * self.row//3):
                    if j < self.col * (2/3):
                        self.maze[i][j] = 1
                        self.reward_states[i][j] = -9
        return self
    
    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if action == 0 and self.state % self.col != (self.col - 1) and self.state + 1 < self.observation_space_dim:     # move right
            if (self.maze[(self.state + 1) // self.row][(self.state + 1) % self.col]) == 0:         
                self.state = self.state + 1
        elif action == 1 and self.state % self.col != 0 and self.state - 1 >= 0:                                        # move left
            if (self.maze[(self.state - 1) // self.row][(self.state - 1) % self.col]) == 0:
                self.state = self.state - 1
        elif action == 2 and self.state < (self.observation_space_dim - self.row):                                      # move down
            if (self.maze[(self.state + self.row) // self.row][(self.state + self.row) % self.col]) == 0:
                self.state = self.state + self.row 
        elif action == 3 and self.state > (self.row - 1):                                                               # move up
            if (self.maze[(self.state - self.row) // self.row][(self.state - self.row) % self.col]) == 0:
                self.state = self.state - self.row
        
        if self.state == self.observation_space_dim - self.row:
            reward = self.success_reward
            done = True
        else:
            reward = self.failed_reward
            done = False
        
        return self.state, reward, done
    
class TwoDimCoordinationMap(TwoDimArrayMap):
    def __init__(self, x_dim, y_dim, action_space_dim=4):
        super().__init__(x_dim, y_dim, action_space_dim)
        self.state = np.array([0, 0])
        self.goal = np.array([self.row - 1, 0])
        self.reward_states[self.goal[0]][self.goal[1]] = self.success_reward    
        
    def reset(self):
        self.state = np.array([0, 0])
        return self.state
    
    def step(self, action):
        if action == 0 and self.state[1] < self.col-1:
            if self.maze[self.state[0]][self.state[1]+1] == 0:
                self.state[1] += 1
        elif action == 1 and self.state[1] > 0:
            if self.maze[self.state[0]][self.state[1]-1] == 0:
                self.state[1] -= 1
        elif action == 2 and self.state[0] < self.row-1:
            if self.maze[self.state[0]+1][self.state[1]] == 0:
                self.state[0] += 1
        elif action == 3 and self.state[0] > 0:
            if self.maze[self.state[0]-1][self.state[1]] == 0:
                self.state[0] -= 1
        
        if (self.state[0] == self.goal[0]) and (self.state[1] == self.goal[1]):
            reward = self.success_reward
            done = True
        else:
            reward = self.failed_reward
            done = False
        return self.state, reward, done