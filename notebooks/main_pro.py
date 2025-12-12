import numpy as np

from core.loader import load_all
from core.correspondences import get_correspondences
from core.pose_estimation import solve_pnp_ransac, Rt_to_matrix, solve_relative_pose_essential, compute_parallax, estimate_rotation_window, solve_pnp
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
    # img_name = "IMG_0171"
    img_name = "IMG_0309"

    tracks_path = f"../data/tracks/{img_name}_tracks/cotracker_superpoint_tracks.npz"
    depth_path  = f"../data/depth/{img_name}_depth/{img_name}_depths.npz"
    intr_path   = "../data/intrinsics.json"
    output_folder = f"../results/{img_name}"

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    print("[INFO] Loading data...")
    tracks, depth_maps, K, intr_meta = load_all(tracks_path, depth_path, intr_path)

    # --------------------------------------------------------
    # TRACK LENGTH FILTER (removes noisy short-lived points)
    # --------------------------------------------------------
    MIN_TRACK_LENGTH = 8
    track_lengths = tracks["visibility"].sum(axis=0)
    long_track_mask = track_lengths >= MIN_TRACK_LENGTH

    tracks["points"]     = tracks["points"][:, long_track_mask, :]
    tracks["visibility"] = tracks["visibility"][:, long_track_mask]
    tracks["num_tracks"] = tracks["points"].shape[1]

    print(f"[INFO] Tracks filtered -> remaining: {tracks['num_tracks']}")

    # --------------------------------------------------------
    # DOWNSAMPLE VIDEO
    # --------------------------------------------------------
    print("[INFO] Downsampling frames by factor 2...")
    step = 2
    tracks = downsample_tracks(tracks, step)
    depth_maps = depth_maps[::step]

    num_frames_tracks = tracks["num_frames"]
    num_frames_depth  = depth_maps.shape[0]
    T = min(num_frames_tracks, num_frames_depth)

    print(f"[INFO] Frames after downsampling: {T}")

    # --------------------------------------------------------
    # RELATIVE POSE ESTIMATION WITH PARALLAX + GROUPING LOGIC
    # --------------------------------------------------------
    relative_poses = []

    PARALLAX_LOW  = 0.004
    PARALLAX_HIGH = 0.005

    print("[INFO] Estimating relative poses...")
    for i in range(T - 1):
        j = i + 1

        # 2D tracks for parallax + E-matrix
        vis_i = tracks["visibility"][i]
        vis_j = tracks["visibility"][j]

        common = np.logical_and(vis_i == 1, vis_j == 1)
        if np.sum(common) < 8:
            relative_poses.append(None)
            continue

        pts2d_i = tracks["points"][i, common]
        pts2d_j = tracks["points"][j, common]

        if pts2d_i.shape[0] < 8 or pts2d_j.shape[0] < 8:
            relative_poses.append(None)
            continue

        # Compute parallax
        parallax = compute_parallax(pts2d_i, pts2d_j, K)

        # ---------------------------------------------
        # MODE 1: PURE ROTATION (NO PARALLAX)
        # ---------------------------------------------
        if parallax < PARALLAX_LOW:
            R = estimate_rotation_window(tracks, K, j, window=5)
            t = np.zeros(3)
            relative_poses.append(Rt_to_matrix(R, t))
            print(f"[{i}->{j}] rotation-only (parallax={parallax:.5f})")
            continue

        # ---------------------------------------------
        # MODE 2: HAS TRANSLATION → USE PnP
        # ---------------------------------------------
        pts3d_i, pts2d_j_3d, _ = get_correspondences(tracks, depth_maps, K, i, j)

        if parallax > PARALLAX_HIGH:
            if pts3d_i is not None and pts3d_i.shape[0] >= 6:
                R, t, ok = solve_pnp_ransac(pts3d_i, pts2d_j_3d, K)
                if ok:
                    relative_poses.append(Rt_to_matrix(R, t))
                    print(f"[{i}->{j}] PnP (parallax={parallax:.5f})")
                    continue

        # ---------------------------------------------
        # MODE 3: HYBRID (AMBIGUOUS PARALLAX)
        # ---------------------------------------------
        R_e, t_e, ok_e = solve_relative_pose_essential(pts2d_i, pts2d_j, K)
        if ok_e:
            # If PnP can refine translation, combine them
            if pts3d_i is not None and pts3d_i.shape[0] >= 6:
                R_p, t_p, ok_p = solve_pnp_ransac(pts3d_i, pts2d_j_3d, K)
                if ok_p:
                    relative_poses.append(Rt_to_matrix(R_e, t_p))
                    print(f"[{i}->{j}] hybrid (parallax={parallax:.5f})")
                    continue

            # fallback: rotation only
            relative_poses.append(Rt_to_matrix(R_e, np.zeros(3)))
            print(f"[{i}->{j}] hybrid-fallback (parallax={parallax:.5f})")
            continue

        # total failure → append None
        print(f"[{i}->{j}] failed (parallax={parallax:.5f})")
        relative_poses.append(None)

    # --------------------------------------------------------
    # BUILD GLOBAL TRAJECTORY
    # --------------------------------------------------------
    print("[INFO] Building global trajectory...")
    global_poses = build_global_trajectory(relative_poses)

    # --------------------------------------------------------
    # SAVE + VISUALIZE
    # --------------------------------------------------------
    np.savez(f"{output_folder}/poses_pro.npz", global_poses=np.array(global_poses))

    plot_trajectory(
        global_poses,
        title="Trajectory",
        save_path=f"{output_folder}/trajectory_pro.png"
    )

    print("[INFO] Pose estimation finished.")


if __name__ == "__main__":
    main()