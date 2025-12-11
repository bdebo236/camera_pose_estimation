import numpy as np
import matplotlib.pyplot as plt

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range)

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])


def plot_trajectory(global_poses, title="Camera Trajectory", save_path="../results/trajectory.png"):
    xs, ys, zs = [], [], []

    for T in global_poses:
        cam_pos = T[:3, 3]
        xs.append(cam_pos[0])
        ys.append(cam_pos[1])
        zs.append(cam_pos[2])

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    fig = plt.figure(figsize=(14, 10))

    # -------------------------
    # 1. Main 3D view
    # -------------------------
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot(xs, ys, zs, "-o", markersize=2)
    set_axes_equal(ax1)
    ax1.set_title(title + " (3D)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # projections remain unchanged...
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(xs, ys, "-o", markersize=2)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y")
    ax2.set_title("XY Projection (Top-down)")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(xs, zs, "-o", markersize=2)
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlabel("X"); ax3.set_ylabel("Z")
    ax3.set_title("XZ Projection")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(ys, zs, "-o", markersize=2)
    ax4.set_aspect("equal", adjustable="box")
    ax4.set_xlabel("Y"); ax4.set_ylabel("Z")
    ax4.set_title("YZ Projection")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()