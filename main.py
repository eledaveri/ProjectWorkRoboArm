from arm import PlanarArm2DOF
from obstacles import make_rect, make_circle, make_polygon
from cspace import ConfigurationSpace
from visualize import plot_cspace, plot_workspace
import numpy as np

def main():
    arm = PlanarArm2DOF([1.0, 1.0])

    # Define some obstacles
    obstacles = [
    make_rect(0.6, 0.7, 0.2, 0.3),
    make_rect(0.8, 0.9, 0.8, 0.9),
    make_circle(-0.2, 0.5, 0.1),
    make_polygon([(-0.5, -0.5), (-0.3, -0.4), (-0.4, -0.2)])
]


    # Plot workspace with arm in a sample configuration
    theta1, theta2 = np.pi/4, np.pi/3
    plot_workspace(arm, theta1, theta2, obstacles)

    # Build C-space
    cspace = ConfigurationSpace(
        arm=arm,
        theta1_range=(0, 2*np.pi),
        theta2_range=(0, 2*np.pi),
        N1=60, N2=60,
        obstacles=obstacles
    )
    cspace.build()

    # Plot C-space
    plot_cspace(cspace)

if __name__ == "__main__":
    main()
