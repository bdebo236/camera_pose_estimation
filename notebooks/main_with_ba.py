import numpy as np

from core.loader import load_all
from core.correspondences import get_correspondences
from core.pose_estimation import solve_pnp, solve_pnp_ransac, Rt_to_matrix
from core.trajectory import build_global_trajectory
from core.viz import plot_trajectory
from core.ba import refine_reconstruction


def downsample_tracks(tracks, step):
    tracks_new = tracks.copy()
    tracks_new["points"] = tracks["points"][::step]
    tracks_new["visibility"] = tracks["visibility"][::step]
    tracks_new["num_frames"] = tracks_new["points"].shape[0]
    return tracks_new


def matrix_to_Rt(T):
    """Extract R and t from 4x4 transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t


def main():
    img_name = "IMG_0112"
    # img_name = "IMG_0171"
    tracks_path = f"../data/tracks/{img_name}_tracks/cotracker_superpoint_tracks.npz"
    depth_path  = f"../data/depth/{img_name}_depth/{img_name}_depths.npz"
    intr_path   = "../data/intrinsics.json"
    output_folder = f"../results/{img_name}"

    # Configuration
    USE_BUNDLE_ADJUSTMENT = True  # Set to False to disable BA
    MIN_TRACK_LENGTH = 8
    DOWNSAMPLE_STEP = 2
    BA_REPROJ_ERROR = 2.0  # Outlier threshold for BA (pixels)

    # Load all data
    print("[INFO] Loading data...")
    tracks, depth_maps, K, intr_meta = load_all(tracks_path, depth_path, intr_path)
    
    # Track Length Filter
    print(f"[INFO] Filtering tracks (min length: {MIN_TRACK_LENGTH})...")
    track_lengths = tracks["visibility"].sum(axis=0) 
    long_track_mask = track_lengths >= MIN_TRACK_LENGTH
    
    original_track_count = tracks["num_tracks"]
    tracks["points"] = tracks["points"][:, long_track_mask, :]
    tracks["visibility"] = tracks["visibility"][:, long_track_mask]
    tracks["num_tracks"] = tracks["points"].shape[1]
    
    print(f"[INFO] Tracks: {original_track_count} -> {tracks['num_tracks']} "
          f"({original_track_count - tracks['num_tracks']} removed)")

    # Downsample frames
    print(f"[INFO] Downsampling frames by factor {DOWNSAMPLE_STEP}...")
    tracks = downsample_tracks(tracks, DOWNSAMPLE_STEP)
    depth_maps = depth_maps[::DOWNSAMPLE_STEP]

    num_frames_tracks = tracks["num_frames"]
    num_frames_depth = depth_maps.shape[0]
    T = min(num_frames_tracks, num_frames_depth)
    print(f"[INFO] Total frames after downsampling: {T}")

    # ========================================================================
    # Step 1: Initial Pose Estimation (Sequential PnP)
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: Initial Pose Estimation (Sequential PnP)")
    print("="*70)
    
    relative_poses = []
    initial_poses = []
    
    # First pose is identity (world frame)
    initial_poses.append((np.eye(3), np.zeros(3)))
    
    for i in range(T - 1):
        j = i + 1

        pts3d_i, pts2d_j, _ = get_correspondences(tracks, depth_maps, K, i, j)
        
        if pts3d_i is None or pts3d_i.shape[0] < 6:
            print(f"[WARN] Frame {i}->{j}: Insufficient correspondences")
            relative_poses.append(None)
            continue

        # Use RANSAC PnP for robustness
        R, t, ok = solve_pnp_ransac(pts3d_i, pts2d_j, K)
        
        if not ok:
            print(f"[WARN] Frame {i}->{j}: PnP failed")
            relative_poses.append(None)
            continue

        relative_poses.append(Rt_to_matrix(R, t))
        
        if (i + 1) % 10 == 0:
            print(f"[INFO] Processed {i+1}/{T-1} frame pairs...")

    # Build initial global trajectory
    global_poses_initial = build_global_trajectory(relative_poses)
    
    # Convert 4x4 matrices to (R, t) tuples for BA
    for T_mat in global_poses_initial:
        R, t = matrix_to_Rt(T_mat)
        initial_poses.append((R, t))
    
    # Trim to match actual number of frames
    initial_poses = initial_poses[:T]
    
    print(f"[INFO] Initial trajectory computed: {len(initial_poses)} poses")

    # ========================================================================
    # Step 2: Bundle Adjustment (Optional)
    # ========================================================================
    if USE_BUNDLE_ADJUSTMENT:
        print("\n" + "="*70)
        print("STEP 2: Bundle Adjustment Refinement")
        print("="*70)
        
        try:
            refined_poses, points_3d, valid_mask = refine_reconstruction(
                tracks=tracks,
                initial_poses=initial_poses,
                K=K,
                max_reproj_error=BA_REPROJ_ERROR,
                verbose=1
            )
            
            # Convert refined poses back to 4x4 matrices
            global_poses_refined = []
            for R, t in refined_poses:
                global_poses_refined.append(Rt_to_matrix(R, t))
            
            # Use refined poses
            global_poses_final = global_poses_refined
            
            # Save refined structure
            np.savez(
                f"{output_folder}/refined_structure.npz",
                points_3d=points_3d,
                valid_mask=valid_mask,
                poses=global_poses_final
            )
            print(f"[INFO] Refined 3D structure saved: {valid_mask.sum()} valid points")
            
        except Exception as e:
            print(f"[ERROR] Bundle adjustment failed: {e}")
            print("[INFO] Falling back to initial poses")
            global_poses_final = global_poses_initial
    else:
        print("\n[INFO] Bundle adjustment disabled, using initial poses")
        global_poses_final = global_poses_initial

    # ========================================================================
    # Step 3: Save and Visualize
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: Saving Results")
    print("="*70)
    
    # Save poses
    np.savez(
        f"{output_folder}/poses_ba.npz", 
        global_poses=np.array(global_poses_final)
    )
    print(f"[INFO] Poses saved to {output_folder}/poses_ba.npz")
    
    # Visualize trajectory
    trajectory_title = "Trajectory (with Bundle Adjustment)" if USE_BUNDLE_ADJUSTMENT else "Trajectory (Initial PnP)"
    plot_trajectory(
        global_poses_final,
        title=trajectory_title,
        save_path=f"{output_folder}/trajectory_ba.png"
    )
    print(f"[INFO] Trajectory plot saved to {output_folder}/trajectory_ba.png")
    
    # Optional: Save comparison if BA was used
    if USE_BUNDLE_ADJUSTMENT:
        plot_trajectory(
            global_poses_initial,
            title="Trajectory (Before BA)",
            save_path=f"{output_folder}/trajectory_before_ba.png"
        )
        print(f"[INFO] Comparison plot saved to {output_folder}/trajectory_before_ba.png")

    print("\n" + "="*70)
    print("✅ Pipeline Complete!")
    print("="*70)


if __name__ == "__main__":
    main()