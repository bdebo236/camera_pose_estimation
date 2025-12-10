import json
import numpy as np

# fix intrinsics if JSON is landscape but video is portrait
def fix_intrinsics_orientation(K, meta_json, orig_h, orig_w):
    json_h = meta_json["height"]
    json_w = meta_json["width"]

    print("json_h,json_w =", json_h, json_w)
    print("orig_h,orig_w =", orig_h, orig_w)
    print("Condition json_h==orig_w ?", json_h == orig_w)
    print("Condition json_w==orig_h ?", json_w == orig_h)

    # detect mismatch JSON says (1440,1920) but actual is (1920,1440)
    if (json_h == orig_w and json_w == orig_h):
        print("[INFO] Detected orientation mismatch. Rotating intrinsics by 90 degrees.")

        fx_old = K[0, 0]
        fy_old = K[1, 1]
        cx_old = K[0, 2]
        cy_old = K[1, 2]

        # after rotation, the actual orientation is portrait: (orig_h, orig_w)
        new_h = orig_h
        new_w = orig_w

        K_rot = np.zeros_like(K)
        K_rot[0, 0] = fy_old
        K_rot[1, 1] = fx_old
        K_rot[0, 2] = new_w - cy_old
        K_rot[1, 2] = cx_old
        K_rot[2, 2] = 1.0

        return K_rot, {"width": new_w, "height": new_h}

    # no mismatch → no rotation
    return K, meta_json


# scale intrinsics for resized frames
def scale_intrinsics(K, scale_x, scale_y):
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x   # fx
    K_scaled[1, 1] *= scale_y   # fy
    K_scaled[0, 2] *= scale_x   # cx
    K_scaled[1, 2] *= scale_y   # cy
    return K_scaled

# load NPZ tracks
def load_tracks(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    keys = data.files

    if "tracks" not in keys:
        raise ValueError("NPZ missing key 'tracks'")

    tracks_arr = data["tracks"]
    visibility = data["visibility"]

    tracks = {
        "points": tracks_arr.astype(np.float32),
        "visibility": visibility.astype(bool),
        "meta": {k: data[k] for k in keys if k not in ["tracks", "visibility"]},
        "num_frames": tracks_arr.shape[0],
        "num_tracks": tracks_arr.shape[1],
    }

    return tracks

# load depth maps
def load_depth(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    depth = data["depths"] if "depths" in data.files else data[data.files[0]]
    return depth.astype(np.float32)

# load intrinsics JSON
def load_intrinsics(json_path):
    with open(json_path, "r") as f:
        intr = json.load(f)

    K = np.array([
        [intr["fx"], 0, intr["cx"]],
        [0, intr["fy"], intr["cy"]],
        [0, 0, 1],
    ], dtype=np.float32)

    meta = {"width": intr["width"], "height": intr["height"]}

    return K, meta

# main unified loader
def load_all(tracks_path, depth_path, intrinsics_path):
    tracks = load_tracks(tracks_path)
    depth_maps = load_depth(depth_path)
    K_json, meta_json = load_intrinsics(intrinsics_path)

    # video resolution
    orig_h = int(tracks["meta"]["original_height"])
    orig_w = int(tracks["meta"]["original_width"])

    # fix portrait vs landscape
    K_fixed, meta_fixed = fix_intrinsics_orientation(
        K_json,
        meta_json,
        orig_h,
        orig_w
    )

    # now scale intrinsics for resized frames
    scale_x = tracks["meta"]["scale_x"]
    scale_y = tracks["meta"]["scale_y"]

    K_resized = scale_intrinsics(K_fixed, scale_x, scale_y)

    # debug print
    print("===== DEBUG K VALUES =====")
    print("K_json =", K_json)
    print("K_fixed =", K_fixed)
    print("K_resized =", K_resized)
    print("==========================")

    return tracks, depth_maps, K_resized, meta_fixed