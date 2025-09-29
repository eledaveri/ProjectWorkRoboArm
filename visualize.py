import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors

def plot_cspace(cspace):
    """Plot the configuration space grid"""
    cmap = mcolors.ListedColormap(["white", "blue", "red"])
    bounds = [0, 1, 2, 3]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(
        cspace.grid.T,
        origin="lower",
        extent=[cspace.theta1_vals[0], cspace.theta1_vals[-1],
                cspace.theta2_vals[0], cspace.theta2_vals[-1]],
        cmap=cmap,
        norm=norm,
        aspect="auto"
    )
    plt.xlabel("θ1 (rad)")
    plt.ylabel("θ2 (rad)")
    plt.title("Configuration Space")
    plt.colorbar(ticks=[0.5, 1.5, 2.5], label="Occupancy", format=lambda x, _: ["Free","Robot","Obstacle"][int(x)])
    plt.show()
    plt.savefig("cspace.png", dpi=300)

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