import numpy as np

# Shared visibility mask
def get_shared_track_mask(tracks, i, j):
    vis = tracks["visibility"]
    if vis is None:
        raise ValueError("Tracks have no visibility matrix")
    return np.logical_and(vis[i], vis[j])

def backproject_point(u, v, depth, fx, fy, cx, cy):
    Z = depth
    if Z <= 0 or not np.isfinite(Z):
        return None # invalid depth

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return np.array([X, Y, Z], dtype=np.float32)

def backproject_points(points_2d, depth_map, K):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    N = points_2d.shape[0]
    
    pts3d = np.zeros((N, 3), dtype=np.float32)
    
    valid_mask = np.ones(N, dtype=bool)
    
    for i in range(N):
        u,v = points_2d[i]
        
        u_i = int(round(u))
        v_i = int(round(v))
        
        # Safe bounds check
        if u_i < 0 or u_i >= depth_map.shape[1] or v_i < 0 or v_i >= depth_map.shape[0]:
            valid_mask[i] = False
            continue
            
        Z = depth_map[v_i,u_i]
        
        if Z <= 0 or not np.isfinite(Z):
            valid_mask[i] = False
            continue
            
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        
        pts3d[i] = [X,Y,Z]

    return pts3d, valid_mask

# 2D → 3D (Depth-based PnP) Correspondences
def get_correspondences(tracks, depth_maps, K, i, j):
    """
    For frame i -> j:
      pts3d_i from depth + tracks[i]
      pts2d_j from tracks[j]
    """
    mask = get_shared_track_mask(tracks, i, j)
    if mask.sum() < 6:
        return None, None, mask

    pts_i = tracks["points"][i][mask]  # (M,2)
    pts_j = tracks["points"][j][mask]  # (M,2)
    depth_i = depth_maps[i]            # (H,W)

    pts3d_i, valid = backproject_points(pts_i, depth_i, K)

    if valid.sum() < 6:
        return None, None, mask

    pts3d_i = pts3d_i[valid]
    pts2d_j = pts_j[valid]

    return pts3d_i, pts2d_j, mask