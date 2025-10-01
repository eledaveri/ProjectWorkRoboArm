import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
from scipy.ndimage import label
import numpy as np

def plot_cspace(cspace):
    """Plot the configuration space grid"""
    cmap = mcolors.ListedColormap(["white", "Blue"])  # 0=free, 1=obstacle
    bounds = [0, 1, 2]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(6,5))
    plt.imshow(
        cspace.grid.T,
        origin="lower",
        extent=[cspace.theta1_vals[0], cspace.theta1_vals[-1],
                cspace.theta2_vals[0], cspace.theta2_vals[-1]],
        cmap=cmap,
        norm=norm,
        aspect="auto"
    )
    plt.xlabel(r"$\theta_1$ (rad)")
    plt.ylabel(r"$\theta_2$ (rad)")
    plt.title("Configuration Space")

    cbar = plt.colorbar(ticks=[0.5, 1.5])
    cbar.ax.set_yticklabels(["Free", "Obstacle"])

    plt.tight_layout()
    plt.savefig("cspace.png", dpi=300)
    plt.show()

def plot_cspace_components(cspace, filename="cspace_components.png"):
    """
    Plot C-space with connected components of free space.

    Args:
        cspace: ConfigurationSpace object
        filename: output image file name
    """

    # Copy the grid
    grid = cspace.grid.copy()

    # Find connected components in free space
    free_space = (grid == 0)
    labeled, num_components = label(free_space)

    print(f"Found {num_components} connected components in free space")

    # Component map, -1 for obstacles
    comp_map = labeled.copy()
    comp_map[grid == 1] = -1

    # Number of bins: obstacles (-1) + components (0..num_components-1)
    num_bins = 1 + num_components

    # Colormap: red for obstacles + colors for components
    cmap = plt.cm.get_cmap("tab20", num_components)
    colors = cmap(np.arange(num_components))
    comp_cmap = mcolors.ListedColormap(["red"] + list(colors))

    # Boundaries for BoundaryNorm: center each color on an integer
    bounds = np.arange(-0.5, num_bins + 0.5, 1)
    norm = mcolors.BoundaryNorm(bounds, comp_cmap.N)

    # Plot
    plt.figure(figsize=(6, 5))
    plt.imshow(
        comp_map.T,  # obstacles = -1, components = 0..num_components-1
        origin="lower",
        extent=[cspace.theta1_vals[0], cspace.theta1_vals[-1],
                cspace.theta2_vals[0], cspace.theta2_vals[-1]],
        cmap=comp_cmap,
        norm=norm,
        aspect="auto"
    )
    plt.xlabel(r"$\theta_1$ (rad)")
    plt.ylabel(r"$\theta_2$ (rad)")
    plt.title("Configuration Space - Connected Components")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()



def plot_workspace(arm, theta1, theta2, obstacles):
    """Plot the workspace with the arm in a given configuration and obstacles"""
    fig, ax = plt.subplots()

    # Arm
    segments = arm.get_segments(theta1, theta2)
    for (p0, p1) in segments:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "bo-", linewidth=3)

    # Obstacles
    for obs in obstacles:
        if hasattr(obs, "bounds"):  # shapely polygon
            xmin, ymin, xmax, ymax = obs.bounds
            patch = Polygon(list(obs.exterior.coords), facecolor="red", alpha=0.4)
            ax.add_patch(patch)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Workspace: robot + obstacles")
    plt.show()
    plt.savefig("workspace.png", dpi=300)

def plot_cspace_path(cspace, path, filename="cspace_path.png"):
    """
    Plot the discretized C-space with a path overlaid.

    Args:
        cspace: ConfigurationSpace object
        path: list of (i,j) states representing the path
        filename: output image file name
    """
    # Base C-space
    cmap = mcolors.ListedColormap(["white", "blue"])  # 0=free, 1=obstacle
    bounds = [0, 1, 2]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(6,5))
    plt.imshow(
        cspace.grid.T,
        origin="lower",
        extent=[cspace.theta1_vals[0], cspace.theta1_vals[-1],
                cspace.theta2_vals[0], cspace.theta2_vals[-1]],
        cmap=cmap,
        norm=norm,
        aspect="auto"
    )

    # Convert path indices (i,j) to theta1/theta2 values
    theta1_path = [cspace.theta1_vals[i] for i, j in path]
    theta2_path = [cspace.theta2_vals[j] for i, j in path]

    # Overlay path
    plt.plot(theta1_path, theta2_path, color="red", marker="o", markersize=3, linewidth=2, label="Path")

    plt.xlabel(r"$\theta_1$ (rad)")
    plt.ylabel(r"$\theta_2$ (rad)")
    plt.title("Configuration Space with Q-learning Path")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_workspace_path(arm, path, cspace, start=None, goal=None, filename="workspace_path.png"):
    """
    Plotta la traiettoria del robot nel workspace (x,y) con start e goal evidenziati.
    
    Args:
        arm: oggetto PlanarArm2DOF
        path: lista di stati (i,j) nel C-space
        cspace: oggetto ConfigurationSpace
        start: stato iniziale (i,j)
        goal: stato goal (i,j)
        filename: nome file immagine di output
    """
    start = start if start else (0, 0)
    goal = goal if goal else (cspace.N1-1, cspace.N2-1)

    plt.figure(figsize=(6,6))
    ax = plt.gca()

    # Disegna ostacoli (Shapely Polygons)
    for obs in cspace.obstacles:
        if isinstance(obs, Polygon):
            x, y = obs.exterior.xy
            ax.fill(x, y, color='k', alpha=0.3)

    # Converti path da (i,j) a θ1, θ2
    theta1_path = [cspace.theta1_vals[i] for i,j in path]
    theta2_path = [cspace.theta2_vals[j] for i,j in path]

    # Calcola coordinate end-effector
    x_path = []
    y_path = []
    for th1, th2 in zip(theta1_path, theta2_path):
        pos = arm.forward_kinematics(th1, th2)
        x_path.append(pos[0])
        y_path.append(pos[1])

    # Traiettoria
    plt.plot(x_path, y_path, 'r-o', markersize=3, linewidth=2, label="Path")

    # Evidenzia start e goal
    start_pos = arm.forward_kinematics(cspace.theta1_vals[start[0]], cspace.theta2_vals[start[1]])
    goal_pos = arm.forward_kinematics(cspace.theta1_vals[goal[0]], cspace.theta2_vals[goal[1]])
    plt.scatter(*start_pos, color='green', s=100, label='Start')
    plt.scatter(*goal_pos, color='blue', s=100, label='Goal')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Workspace Path")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()