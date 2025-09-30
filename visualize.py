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
    ax.set_title("Workspace: braccio + ostacoli")
    plt.show()
    plt.savefig("workspace.png", dpi=300)