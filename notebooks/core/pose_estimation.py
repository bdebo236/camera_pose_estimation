import numpy as np
import cv2

def solve_pnp(pts3d, pts2d, K):
    """
    Robust PnP with RANSAC.
    Automatically rejects outlier correspondences.
    """

    if pts3d.shape[0] < 6:
        return None, None, False

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d.astype(np.float32),
        pts2d.astype(np.float32),
        K.astype(np.float32),
        distCoeffs=None,
        reprojectionError=8.0,
        confidence=0.99,
        iterationsCount=200
    )

    # RANSAC failed if:
    if (not success) or (inliers is None) or (len(inliers) < 6):
        return None, None, False

    # Refine with only the inlier points
    inliers = inliers[:,0]
    success, rvec, tvec = cv2.solvePnP(
        pts3d[inliers],
        pts2d[inliers],
        K,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None, False

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    return R, t, True


def Rt_to_matrix(R, t):
    """
    Convert R (3x3) and t (3,) into a 4x4 SE(3) transform matrix.
    """
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T