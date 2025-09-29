import numpy as np

class PlanarArm2DOF:
    def __init__(self, link_lengths):
        self.l1, self.l2 = link_lengths

    def forward_kinematics(self, theta1, theta2):
        """Restituisce (x,y) dell’end-effector"""
        x = self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2)
        y = self.l1 * np.sin(theta1) + self.l2 * np.sin(theta1 + theta2)
        return np.array([x, y])

    def get_segments(self, theta1, theta2):
        """Restituisce i segmenti [(p0,p1), (p1,p2)] del braccio"""
        p0 = np.array([0.0, 0.0])
        p1 = np.array([self.l1 * np.cos(theta1), self.l1 * np.sin(theta1)])
        p2 = self.forward_kinematics(theta1, theta2)
        return [(p0, p1), (p1, p2)]
