import numpy as np
import cv2

def solve_pnp_ransac(pts3d, pts2d, K):
    if pts3d.shape[0] < 6:
        # Need at least 6 points for RANSAC to be stable, 
        # though PnP technically needs 4 (P3P) or 3 (P3P + additional constraints).
        return None, None, False

    # 1. Solve PnP using RANSAC to find inliers
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d.astype(np.float32),
        pts2d.astype(np.float32),
        K.astype(np.float32),
        distCoeffs=None,
        reprojectionError=8.0,
        confidence=0.999,
        iterationsCount=500
    )

    # RANSAC failed if:
    if (not success) or (inliers is None) or (len(inliers) < 6):
        return None, None, False

    # 2. Refine with only the inlier points using an iterative solver
    inliers = inliers[:, 0]
    success, rvec, tvec = cv2.solvePnP(
        pts3d[inliers],
        pts2d[inliers],
        K,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None, False

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    return R, t, True

def solve_pnp(pts3d, pts2d, K, method=cv2.SOLVEPNP_ITERATIVE):
    # PnP needs a minimum of 4 non-coplanar points for a unique solution 
    # (or 3 for P3P, but SOLVEPNP_ITERATIVE needs 4+).
    if pts3d.shape[0] < 4:
        return None, None, False

    success, rvec, tvec = cv2.solvePnP(
        pts3d.astype(np.float32),
        pts2d.astype(np.float32),
        K.astype(np.float32),
        distCoeffs=None,
        flags=method
    )

    if not success:
        return None, None, False

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    return R, t, True

# --- Utility Function (Kept as is) ---
def Rt_to_matrix(R, t):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def compute_parallax(pts2d_i, pts2d_j, K):
    """
    Returns median normalized parallax between two frames.
    Small value → almost pure rotation.
    """
    pts_i_norm = cv2.undistortPoints(
        pts2d_i.astype(np.float32), K.astype(np.float32), None
    ).reshape(-1,2)

    pts_j_norm = cv2.undistortPoints(
        pts2d_j.astype(np.float32), K.astype(np.float32), None
    ).reshape(-1,2)

    diff = pts_i_norm - pts_j_norm
    parallax = np.linalg.norm(diff, axis=1)
    return np.median(parallax)

def solve_relative_pose_essential(pts2d_i, pts2d_j, K):
    """
    Estimate rotation-only relative pose using essential matrix.
    Translation is unreliable if parallax is low → return t = zero.
    """
    E, inliers = cv2.findEssentialMat(
        pts2d_i.astype(np.float32),
        pts2d_j.astype(np.float32),
        K.astype(np.float32),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None:
        return None, None, False

    _, R, t, _ = cv2.recoverPose(
        E,
        pts2d_i.astype(np.float32),
        pts2d_j.astype(np.float32),
        K.astype(np.float32)
    )

    return R, t.reshape(3), True

def estimate_rotation_window(tracks, K, i, window=5):
    """
    Compute a stable rotation for frame i using frames [i-window, ..., i-1].
    Returns averaged rotation matrix.
    """
    Rs = []

    for k in range(max(0, i - window), i):

        # shared tracks only
        vis_k = tracks["visibility"][k]
        vis_i = tracks["visibility"][i]
        common = np.logical_and(vis_k == 1, vis_i == 1)

        if np.sum(common) < 8:
            continue

        pts2d_k = tracks["points"][k, common]
        pts2d_i = tracks["points"][i, common]

        # essential matrix
        R, t, ok = solve_relative_pose_essential(pts2d_k, pts2d_i, K)
        if ok:
            Rs.append(R)

    if len(Rs) == 0:
        return np.eye(3)

    # average all rotations using SVD
    R_stack = sum(Rs)
    U, _, Vt = np.linalg.svd(R_stack)
    R_avg = U @ Vt
    return R_avg