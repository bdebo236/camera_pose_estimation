import numpy as np
import cv2
import json

from core.trajectory import build_global_trajectory
from core.viz import plot_trajectory


# ========== 2D Pose Estimation Functions ==========

def estimate_relative_pose_2d(pts2d_i, pts2d_j, K):
    """
    Estimate relative pose between two frames using Essential Matrix.
    This is the basic 2D-only approach without depth information.
    
    NOTE: Translation scale is ARBITRARY! 
    """
    if pts2d_i.shape[0] < 8:
        return None, None, False
    
    # Find Essential Matrix using RANSAC
    E, mask = cv2.findEssentialMat(
        pts2d_i.astype(np.float32),
        pts2d_j.astype(np.float32),
        K.astype(np.float32),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )
    
    if E is None:
        return None, None, False
    
    # Recover pose from Essential Matrix
    _, R, t, mask_pose = cv2.recoverPose(
        E,
        pts2d_i.astype(np.float32),
        pts2d_j.astype(np.float32),
        K.astype(np.float32),
        mask=mask
    )
    
    if mask_pose is not None and np.sum(mask_pose) < 8:
        return None, None, False
    
    return R, t.reshape(3), True


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
    intr_path   = "../data/intrinsics.json"
    output_folder = f"../results/{img_name}_2d"

    # load only tracks and intrinsics (no depth needed for 2D-only!)
    print("[INFO] Loading data...")
    import json
    
    # Load tracks
    tracks_data = np.load(tracks_path)
    tracks = {
        "points": tracks_data["tracks"],
        "visibility": tracks_data["visibility"],
        "num_frames": tracks_data["tracks"].shape[0],
        "num_tracks": tracks_data["tracks"].shape[1]
    }
    
    # Load intrinsics
    with open(intr_path, 'r') as f:
        intr_meta = json.load(f)
    
    K = np.array([
        [intr_meta["fx"], 0, intr_meta["cx"]],
        [0, intr_meta["fy"], intr_meta["cy"]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Track Length Filter
    MIN_TRACK_LENGTH = 8
    track_lengths = tracks["visibility"].sum(axis=0) 
    long_track_mask = track_lengths >= MIN_TRACK_LENGTH
    
    tracks["points"] = tracks["points"][:, long_track_mask, :]
    tracks["visibility"] = tracks["visibility"][:, long_track_mask]
    tracks["num_tracks"] = tracks["points"].shape[1]
    
    print(f"[INFO] Tracks filtered: {len(long_track_mask) - tracks['num_tracks']} removed. Remaining: {tracks['num_tracks']}")

    # downsample frames
    print("[INFO] Downsampling frames by factor 2...")
    step = 2
    tracks = downsample_tracks(tracks, step)

    T = tracks["num_frames"]
    print(f"[INFO] Frames after downsampling: {T}")

    # Estimate relative poses using 2D correspondences only
    relative_poses = []
    for i in range(T - 1):
        j = i + 1
        
        # Get 2D-2D correspondences between frames i and j
        vis_i = tracks["visibility"][i]
        vis_j = tracks["visibility"][j]
        common = np.logical_and(vis_i == 1, vis_j == 1)
        
        if np.sum(common) < 8:
            print(f"[WARN] Frame {i}->{j}: Not enough correspondences ({np.sum(common)})")
            relative_poses.append(None)
            continue
        
        pts2d_i = tracks["points"][i, common]
        pts2d_j = tracks["points"][j, common]
        
        # Estimate relative pose using Essential Matrix
        R, t, ok = estimate_relative_pose_2d(pts2d_i, pts2d_j, K)
        
        if not ok:
            print(f"[WARN] Frame {i}->{j}: Pose estimation failed")
            relative_poses.append(None)
            continue
        
        # Create 4x4 transformation matrix
        T_rel = np.eye(4, dtype=np.float32)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t
        
        relative_poses.append(T_rel)

    # build global trajectory
    global_poses = build_global_trajectory(relative_poses)

    # save + visualize trajectory
    np.savez(f"{output_folder}/poses_2d.npz", global_poses=np.array(global_poses))
    plot_trajectory(global_poses,
                    title="2D-Only Trajectory (No Depth - Arbitrary Scale)",
                    save_path=f"{output_folder}/trajectory_2d.png")

    print("[INFO] 2D pose estimation finished.")
    print("[NOTE] Scale is ARBITRARY - not in real-world units!")


if __name__ == "__main__":
    main()