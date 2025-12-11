"""
Bundle Adjustment Module

Usage in main.py:
    from ba import refine_reconstruction
    
    # After computing initial poses
    refined_poses, refined_points_3d = refine_reconstruction(
        tracks, initial_poses, K, max_reproj_error=2.0
    )
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


# ============================================================================
# Pose Parameterization
# ============================================================================

def pose_to_vector(R, t):
    """Convert rotation matrix and translation to 6D parameter vector."""
    rvec = Rotation.from_matrix(R).as_rotvec()
    return np.concatenate([rvec, t.ravel()])


def vector_to_pose(vec):
    """Convert 6D parameter vector back to R, t."""
    rvec = vec[:3]
    t = vec[3:6]
    R = Rotation.from_rotvec(rvec).as_matrix()
    return R, t


# ============================================================================
# Triangulation
# ============================================================================

def triangulate_track(track_points, track_vis, poses, K):
    """
    Triangulate a single track using DLT.
    
    Args:
        track_points: (n_frames, 2) array of 2D observations
        track_vis: (n_frames,) boolean array of visibility
        poses: list of (R, t) tuples
        K: (3, 3) intrinsic matrix
    
    Returns:
        X: (3,) 3D point in world coordinates, or None if triangulation fails
    """
    # Convert boolean to float if needed
    if track_vis.dtype == bool:
        frames_visible = np.where(track_vis)[0]
    else:
        frames_visible = np.where(track_vis > 0.5)[0]
    
    if len(frames_visible) < 2:
        return None
    
    # Ensure we don't exceed available poses
    frames_visible = frames_visible[frames_visible < len(poses)]
    
    if len(frames_visible) < 2:
        return None
    
    # Build linear system A*X = 0
    A = []
    for frame_idx in frames_visible:
        R, t = poses[frame_idx]
        P = K @ np.hstack([R, t.reshape(3, 1)])
        
        x, y = track_points[frame_idx]
        
        # Cross product equations
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    
    A = np.array(A)
    
    # Solve using SVD
    try:
        _, _, Vt = np.linalg.svd(A)
        X_homo = Vt[-1]
        
        # Check for valid homogeneous coordinate
        if np.abs(X_homo[3]) < 1e-8:
            return None
            
        X = X_homo[:3] / X_homo[3]
        
        # Sanity check: point should be in front of cameras
        for frame_idx in frames_visible:
            R, t = poses[frame_idx]
            X_cam = R @ X + t
            if X_cam[2] <= 0:  # Behind camera
                return None
        
        return X
    except Exception as e:
        return None


def triangulate_all_tracks(tracks, poses, K):
    """
    Triangulate all tracks.
    
    Returns:
        points_3d: (n_tracks, 3) array
        valid_mask: (n_tracks,) boolean array indicating successful triangulation
    """
    n_tracks = tracks['points'].shape[1]
    n_frames = min(tracks['points'].shape[0], len(poses))
    
    points_3d = []
    valid_mask = []
    
    print(f"[BA] Triangulating {n_tracks} tracks across {n_frames} frames...")
    
    for track_idx in range(n_tracks):
        # Extract track data for available frames only
        track_points = tracks['points'][:n_frames, track_idx, :]
        track_vis = tracks['visibility'][:n_frames, track_idx]
        
        X = triangulate_track(track_points, track_vis, poses, K)
        
        if X is not None:
            points_3d.append(X)
            valid_mask.append(True)
        else:
            points_3d.append(np.array([0, 0, 0]))
            valid_mask.append(False)
        
        # Progress indicator
        if (track_idx + 1) % 500 == 0:
            print(f"[BA]   Processed {track_idx + 1}/{n_tracks} tracks...")
    
    return np.array(points_3d), np.array(valid_mask)


# ============================================================================
# Bundle Adjustment
# ============================================================================

def project_points(X_3d, R, t, K):
    """Project 3D points to 2D using camera parameters."""
    # Transform to camera coordinates
    X_cam = R @ X_3d.T + t.reshape(3, 1)
    
    # Project to normalized image plane
    x_norm = X_cam[:2] / (X_cam[2] + 1e-8)
    
    # Apply intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = fx * x_norm[0] + cx
    v = fy * x_norm[1] + cy
    
    return np.vstack([u, v]).T


def bundle_adjustment_residuals(params, n_cameras, n_points, camera_indices, 
                                point_indices, points_2d, K, fix_first_camera):
    """
    Compute residuals for bundle adjustment.
    
    params: [camera_params (6 * n_cameras_opt), points_3d (3 * n_points)]
    """
    n_cameras_opt = n_cameras - 1 if fix_first_camera else n_cameras
    
    camera_params = params[:n_cameras_opt * 6].reshape((n_cameras_opt, 6))
    points_3d = params[n_cameras_opt * 6:].reshape((n_points, 3))
    
    residuals = []
    
    for i, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
        R, t = vector_to_pose(camera_params[cam_idx])
        X = points_3d[pt_idx]
        
        # Project
        projected = project_points(X.reshape(1, 3), R, t, K)
        
        # Residual
        residual = projected[0] - points_2d[i]
        residuals.extend(residual)
    
    return np.array(residuals)


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    """Create sparsity structure for Jacobian."""
    m = camera_indices.size * 2  # 2 residuals per observation
    n = n_cameras * 6 + n_points * 3
    
    A = lil_matrix((m, n), dtype=int)
    
    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1
    
    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1
    
    return A


def run_bundle_adjustment(poses, points_3d, valid_points, tracks, K, 
                         max_reproj_error=2.0, fix_first_camera=True, 
                         verbose=1):
    """
    Run sparse bundle adjustment.
    
    Args:
        poses: List of (R, t) tuples for each camera
        points_3d: (N, 3) initial 3D points
        valid_points: (N,) boolean mask of valid points
        tracks: Dict with 'points' (frames, tracks, 2) and 'visibility'
        K: (3, 3) camera intrinsics
        max_reproj_error: Outlier threshold in pixels
        fix_first_camera: If True, fix first camera pose (recommended)
        verbose: 0=silent, 1=progress, 2=detailed
    
    Returns:
        optimized_poses: List of (R, t) tuples
        optimized_points: (N, 3) array
        inlier_mask: Boolean mask of inlier observations
    """
    n_cameras = len(poses)
    n_points = valid_points.sum()
    
    # Map from original track indices to optimized point indices
    track_to_point = np.full(len(valid_points), -1, dtype=int)
    track_to_point[valid_points] = np.arange(n_points)
    
    # Build observation lists
    camera_indices = []
    point_indices = []
    points_2d = []
    original_track_indices = []
    
    for cam_idx in range(n_cameras):
        vis_mask = tracks['visibility'][cam_idx]
        visible_tracks = np.where(vis_mask)[0]
        
        for track_idx in visible_tracks:
            if valid_points[track_idx]:
                pt_idx = track_to_point[track_idx]
                camera_indices.append(cam_idx)
                point_indices.append(pt_idx)
                points_2d.append(tracks['points'][cam_idx, track_idx])
                original_track_indices.append(track_idx)
    
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    points_2d = np.array(points_2d)
    
    if verbose >= 1:
        print(f"[BA] {n_cameras} cameras, {n_points} points, {len(points_2d)} observations")
    
    # Initialize parameters
    if fix_first_camera:
        camera_params = np.array([pose_to_vector(R, t) for R, t in poses[1:]])
        # Adjust camera indices
        camera_indices_opt = camera_indices - 1
        valid_obs = camera_indices > 0
        camera_indices_opt = camera_indices_opt[valid_obs]
        point_indices_opt = point_indices[valid_obs]
        points_2d_opt = points_2d[valid_obs]
    else:
        camera_params = np.array([pose_to_vector(R, t) for R, t in poses])
        camera_indices_opt = camera_indices
        point_indices_opt = point_indices
        points_2d_opt = points_2d
        valid_obs = np.ones(len(camera_indices), dtype=bool)
    
    # Select valid 3D points
    points_3d_opt = points_3d[valid_points]
    
    # Flatten parameters
    x0 = np.hstack([camera_params.ravel(), points_3d_opt.ravel()])
    
    n_cameras_opt = len(camera_params)
    
    # Sparsity structure
    A = bundle_adjustment_sparsity(
        n_cameras_opt, n_points, 
        camera_indices_opt, 
        point_indices_opt
    )
    
    if verbose >= 1:
        print(f"[BA] Optimizing {len(x0)} parameters...")
    
    # Optimize
    res = least_squares(
        bundle_adjustment_residuals,
        x0,
        jac_sparsity=A,
        verbose=(2 if verbose >= 2 else 0),
        x_scale='jac',
        ftol=1e-4,
        method='trf',
        max_nfev=100,
        args=(n_cameras_opt, n_points, 
              camera_indices_opt, 
              point_indices_opt, 
              points_2d_opt, K, False)
    )
    
    # Extract optimized parameters
    camera_params_opt = res.x[:n_cameras_opt * 6].reshape((n_cameras_opt, 6))
    points_3d_opt = res.x[n_cameras_opt * 6:].reshape((n_points, 3))
    
    # Reconstruct poses
    if fix_first_camera:
        optimized_poses = [poses[0]]
        for cam_params in camera_params_opt:
            R, t = vector_to_pose(cam_params)
            optimized_poses.append((R, t))
    else:
        optimized_poses = []
        for cam_params in camera_params_opt:
            R, t = vector_to_pose(cam_params)
            optimized_poses.append((R, t))
    
    # Reconstruct full points_3d array
    optimized_points_full = points_3d.copy()
    optimized_points_full[valid_points] = points_3d_opt
    
    # Compute final reprojection errors
    final_residuals = np.abs(res.fun).reshape(-1, 2)
    reproj_errors = np.linalg.norm(final_residuals, axis=1)
    inlier_mask = reproj_errors < max_reproj_error
    
    if verbose >= 1:
        print(f"[BA] Complete: {inlier_mask.sum()}/{len(inlier_mask)} inliers")
        print(f"[BA] Mean reprojection error: {reproj_errors[inlier_mask].mean():.2f}px")
        print(f"[BA] Cost: {res.cost:.2e} -> {res.fun.sum():.2e}")
    
    return optimized_poses, optimized_points_full, inlier_mask


# ============================================================================
# Main Interface
# ============================================================================

def refine_reconstruction(tracks, initial_poses, K, max_reproj_error=2.0, 
                         verbose=1):
    """
    Refine camera poses and 3D structure using bundle adjustment.
    
    This is the main function to call from main.py.
    
    Args:
        tracks: Dict with 'points' (n_frames, n_tracks, 2) and 'visibility'
        initial_poses: List of (R, t) tuples from PnP
        K: (3, 3) camera intrinsic matrix
        max_reproj_error: Outlier threshold in pixels
        verbose: 0=silent, 1=progress, 2=detailed
    
    Returns:
        refined_poses: List of (R, t) tuples
        points_3d: (n_tracks, 3) refined 3D points
        valid_mask: (n_tracks,) boolean mask of successfully optimized points
    """
    if verbose >= 1:
        print("\n" + "="*60)
        print("Bundle Adjustment Refinement")
        print("="*60)
    
    # Step 1: Triangulate initial 3D points
    if verbose >= 1:
        print("[BA] Step 1/2: Triangulating tracks...")
    
    points_3d, valid_mask = triangulate_all_tracks(tracks, initial_poses, K)
    
    if verbose >= 1:
        print(f"[BA] Triangulated {valid_mask.sum()}/{len(valid_mask)} tracks")
    
    if valid_mask.sum() < 10:
        print("[BA] ERROR: Too few valid triangulated points. Skipping BA.")
        return initial_poses, points_3d, valid_mask
    
    # Step 2: Run bundle adjustment
    if verbose >= 1:
        print("[BA] Step 2/2: Running bundle adjustment...")
    
    refined_poses, refined_points, inlier_mask = run_bundle_adjustment(
        initial_poses, 
        points_3d, 
        valid_mask, 
        tracks, 
        K,
        max_reproj_error=max_reproj_error,
        fix_first_camera=True,
        verbose=verbose
    )
    
    if verbose >= 1:
        print("="*60 + "\n")
    
    return refined_poses, refined_points, valid_mask