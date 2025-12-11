import numpy as np
import matplotlib.pyplot as plt

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
    ax1.set_title(title + " (3D)")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

    # -------------------------
    # 2. XY projection (top-down)
    # -------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(xs, ys, "-o", markersize=2)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y")
    ax2.set_title("XY Projection (Top-down)")

    # -------------------------
    # 3. XZ projection
    # -------------------------
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(xs, zs, "-o", markersize=2)
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlabel("X"); ax3.set_ylabel("Z")
    ax3.set_title("XZ Projection")

    # -------------------------
    # 4. YZ projection
    # -------------------------
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(ys, zs, "-o", markersize=2)
    ax4.set_aspect("equal", adjustable="box")
    ax4.set_xlabel("Y"); ax4.set_ylabel("Z")
    ax4.set_title("YZ Projection")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()