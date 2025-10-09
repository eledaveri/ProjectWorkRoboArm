from arm import PlanarArm2DOF
from obstacles import make_rect, make_circle, make_polygon
from cspace import ConfigurationSpace
from visualize import plot_cspace, plot_workspace, plot_cspace_components, plot_cspace_path, plot_workspace_path, animate_training_path
from qlearning import QLearning2DOF
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
    plot_cspace_components(cspace)

    # Q-learning# Q-learning
    #start = (0, 0)
    #goal = (cspace.N1-1, cspace.N2-1)
    start = (40, 5)
    goal = (55, 50)
    ql = QLearning2DOF(
        cspace, 
        start=start, 
        goal=goal,
        alpha=0.1,        # Learning rate
        gamma=0.95,       # Discount factor (vicino a 1 per percorsi lunghi)
        epsilon=0.9       # Esplorazione iniziale alta
        )

    ql.train(
        num_episodes=5000,   # Più episodi perché molti termineranno presto
        max_steps=500,
        verbose=True
    )

    # Percorso trovato
    path = ql.get_path()
    print("Learned path:", path)
    animate_training_path(arm, path, cspace, obstacles, "robot_motion.gif")
    
    # Plotta workspace
    plot_workspace_path(arm, path, cspace, start=start, goal=goal)



if __name__ == "__main__":
    main()
