from arm import PlanarArm2DOF
from obstacles import make_rect, make_circle, make_polygon
from cspace import ConfigurationSpace
from visualize import plot_cspace, plot_workspace, plot_cspace_components, plot_workspace_path, animate_training_path
from qlearning import QLearning2DOF
import numpy as np

def main():
    arm = PlanarArm2DOF([1.0, 1.0])

    # Define some obstacles
    obstacles = [
    make_rect(0.6, 0.7, 0.2, 0.3),
    make_rect(0.8, 0.9, 0.8, 0.9),
    make_circle(-0.2, 0.5, 0.1),
    make_polygon([(-0.6, -0.6), (-0.1, -0.9), (-0.4, -0.2)])
    ]

    # Start and goal in the same connected component
    start = (5,5)
    goal = (130, 130)
    #goal = (140, 25)
    #start  = (115, 60)

 
    # Start and goal in different connected components (to test)
    #start = (15, 45)
    #goal = (55, 50)

    # Build C-space
    cspace = ConfigurationSpace(
        arm=arm,
        theta1_range=(0, 2*np.pi),
        theta2_range=(0, 2*np.pi),
        N1=150, N2=150,
        obstacles=obstacles
    )
    cspace.build()
    theta1 = cspace.theta1_vals[start[0]]
    theta2 = cspace.theta2_vals[start[1]]
    
    # Plot C-space
    plot_cspace(cspace)
    plot_cspace_components
    plot_cspace_components(cspace, start=start, goal=goal, filename="cspace_components.png")
    plot_workspace(arm, theta1, theta2, obstacles, start=start, goal=goal, cspace=cspace, filename="workspace_periodical_theta.png")
    # Q-learning
    ql = QLearning2DOF(
        cspace, 
        start=start, 
        goal=goal,
        alpha=0.1,        # Learning rate
        gamma=0.95,       # Discount factor 
        epsilon=0.9       # Initial exploration rate
        )

    ql.train(
        num_episodes=7500,   
        max_steps=500,
        verbose=True
    )

    # Get learned path
    path = ql.get_path()
    print("Learned path:", path)
    animate_training_path(arm, path, cspace, obstacles, start=start, goal=goal, filename="training_periodical_theta.gif")
    
    # Plot workspace
    plot_workspace_path(arm, path, cspace, start=start, goal=goal, filename="workspace_periodical_theta_path.png")


if __name__ == "__main__":
    main()
