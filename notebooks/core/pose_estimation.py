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