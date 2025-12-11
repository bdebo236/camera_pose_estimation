import numpy as np

def compose_T(T_prev, T_delta):
    return T_prev @ T_delta

def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]

    R_inv = R.T
    t_inv = -R_inv @ t

    T_inv = np.eye(4, dtype=np.float32)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def build_global_trajectory(relative_poses):
    global_poses = []
    T_global = np.eye(4, dtype=np.float32)
    global_poses.append(T_global.copy())

    for T_delta in relative_poses:
        if T_delta is None:
            # Propagate previous pose unchanged (or handle differently)
            global_poses.append(T_global.copy())
            continue

        T_global = compose_T(T_global, T_delta)
        global_poses.append(T_global.copy())

    return global_poses