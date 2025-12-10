import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(global_poses, title="Camera Trajectory", save_path="../results/trajectory.png"):
    xs, ys, zs = [], [], []

    for T in global_poses:
        cam_pos = T[:3, 3]
        xs.append(cam_pos[0])
        ys.append(cam_pos[1])
        zs.append(cam_pos[2])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(xs, ys, zs, "-o", markersize=2)
    ax.set_title(title)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_points_3d(points, title="3D Points", save_path="../results/points_3d.png"):
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(xs, ys, zs, s=1)
    ax.set_title(title)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()