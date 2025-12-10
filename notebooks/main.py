import numpy as np

from core.loader import load_all
from core.correspondences import get_correspondences
from core.pose_estimation import solve_pnp, Rt_to_matrix
from core.trajectory import build_global_trajectory
from core.viz import plot_trajectory

def downsample_tracks(tracks, step):
    tracks_new = tracks.copy()
    tracks_new["points"] = tracks["points"][::step]
    tracks_new["visibility"] = tracks["visibility"][::step]
    tracks_new["num_frames"] = tracks_new["points"].shape[0]
    return tracks_new


def main():
    # img_name = "IMG_0112"
    img_name = "IMG_0171"
    tracks_path = f"../data/tracks/{img_name}_cotracker_superpoint/cotracker_superpoint_tracks.npz"
    depth_path  = f"../data/depth/{img_name}_depth/{img_name}_depths.npz"
    intr_path   = "../data/intrinsics.json"
    output_folder = f"../results/{img_name}"

    # load all data (now returns corrected rotated + scaled K)
    print("[INFO] Loading data...")
    tracks, depth_maps, K, intr_meta = load_all(tracks_path, depth_path, intr_path)

    # downsample frames
    print("[INFO] Downsampling frames by factor 2...")
    step = 2
    tracks = downsample_tracks(tracks, step)
    depth_maps = depth_maps[::step]

    T = tracks["num_frames"]
    print(f"[INFO] Frames after downsampling: {T}")

    relative_poses = []
    for i in range(T - 1):
        j = i + 1

        pts3d_i, pts2d_j, _ = get_correspondences(tracks, depth_maps, K, i, j)
        if pts3d_i is None or pts3d_i.shape[0] < 6:
            relative_poses.append(None)
            continue

        R, t, ok = solve_pnp(pts3d_i, pts2d_j, K)
        if not ok:
            relative_poses.append(None)
            continue

        relative_poses.append(Rt_to_matrix(R, t))

    # build global trajectory
    global_poses = build_global_trajectory(relative_poses)

    # save + visualize trajectory
    np.savez(f"{output_folder}/poses.npz", global_poses=np.array(global_poses))
    plot_trajectory(global_poses,
                    title="Initial PnP Trajectory (No BA)",
                    save_path=f"{output_folder}/trajectory.png")

    print("[INFO] Pose estimation finished (no BA).")


if __name__ == "__main__":
    main()