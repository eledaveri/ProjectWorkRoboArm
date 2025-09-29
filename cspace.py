import numpy as np
from arm import PlanarArm2DOF
from obstacles import check_collision

class ConfigurationSpace:
    def __init__(self, arm, theta1_range, theta2_range, N1, N2, obstacles=[]):
        self.arm = arm
        self.theta1_vals = np.linspace(theta1_range[0], theta1_range[1], N1)
        self.theta2_vals = np.linspace(theta2_range[0], theta2_range[1], N2)
        self.grid = np.zeros((N1, N2), dtype=int)  # 0=free, 1=occupied
        self.obstacles = obstacles
        self.N1, self.N2 = N1, N2

    def build(self):
        for i, th1 in enumerate(self.theta1_vals):
            for j, th2 in enumerate(self.theta2_vals):
                segments = self.arm.get_segments(th1, th2)
                if check_collision(segments, self.obstacles):
                    self.grid[i, j] = 1
        return self.grid
