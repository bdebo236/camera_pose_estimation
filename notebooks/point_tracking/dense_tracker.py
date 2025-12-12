import os
import sys
import numpy as np
import torch
import cv2

# CoTracker repo
sys.path.append(os.path.expanduser("~/co-tracker"))

from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

# IMG_NAME       = "IMG_0112"
IMG_NAME       = "IMG_0309"
# IMG_NAME       = "IMG_0171"
VIDEO_PATH      = f"../data/videos/{IMG_NAME}.MOV"
SP_KPTS_PATH    = f"../data/tracks/{IMG_NAME}_tracks/superpoint_kpts.npz"
CHECKPOINT      = os.path.expanduser("~/co-tracker/checkpoints/scaled_offline.pth")
SAVE_DIR        = f"../data/tracks/{IMG_NAME}_tracks/"
QUERY_FRAME     = 20        # like grid_query_frame=20
MAX_POINTS      = 500       # limit for stability / speed

# Original resolution (where SuperPoint was run)
H0, W0 = 1440, 1920
# Downsampled resolution used for CoTracker demo
Hr, Wr = 720, 960

# Visibility threshold for killing tracks
VIS_THRESH = 0.7

# Parameters for adding new points for new objects
NEW_POINT_INTERVAL = 30    # frames between new SuperPoint injections
DIST_THRESH = 10.0         # px in resized space to consider "too close" to an existing track
MAX_NEW_POINTS_PER_INJECTION = 200

os.makedirs(SAVE_DIR, exist_ok=True)

# Load video and resize
video_np = read_video_from_path(VIDEO_PATH)  # (T, H, W, C)
T = video_np.shape[0]

video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float()  # (1, T, C, H, W)
B, T_check, C, H, W = video.shape
assert T == T_check

# resize spatial dims
video = video.reshape(B * T, C, H, W)
video = torch.nn.functional.interpolate(video, size=(Hr, Wr), mode="bilinear")
video = video.reshape(B, T, C, Hr, Wr)

if torch.cuda.is_available():
    video = video.cuda()

print("Video shape for CoTracker:", video.shape)  # (1, T, 3, Hr, Wr)

# Load SuperPoint keypoints and pick those from QUERY_FRAME
sp = np.load(SP_KPTS_PATH, allow_pickle=True)
kpts_all = sp["keypoints"]  # list length T, each (N_i, 2)

kpts_q = kpts_all[QUERY_FRAME]  # (N, 2) for this frame
if kpts_q.shape[0] > MAX_POINTS:
    # random subset to avoid overkill
    idx = np.random.choice(kpts_q.shape[0], size=MAX_POINTS, replace=False)
    kpts_q = kpts_q[idx]

# scale keypoints from original (W0, H0) to resized (Wr, Hr)
scale_x = Wr / W0
scale_y = Hr / H0
kpts_q_resized = np.zeros_like(kpts_q, dtype=np.float32)
kpts_q_resized[:, 0] = kpts_q[:, 0] * scale_x  # x
kpts_q_resized[:, 1] = kpts_q[:, 1] * scale_y  # y

# build CoTracker queries tensor
# format: [time, x, y] per point
Nq = kpts_q_resized.shape[0]
queries_np = np.zeros((Nq, 3), dtype=np.float32)
queries_np[:, 0] = float(QUERY_FRAME)           # time index
queries_np[:, 1:] = kpts_q_resized              # x, y

queries = torch.from_numpy(queries_np)          # (Nq, 3)
if torch.cuda.is_available():
    queries = queries.cuda()

# add batch dim for model
queries = queries[None]  # (1, Nq, 3)

# run CoTracker with backward=True (like grid_query_20_backward)
model = CoTrackerPredictor(checkpoint=CHECKPOINT)
if torch.cuda.is_available():
    model = model.cuda()

with torch.no_grad():
    pred_tracks, pred_visibility = model(
        video,
        queries=queries,
        backward_tracking=True
    )

# pred_tracks: (B, T, Nq, 2) in resized coordinates
# pred_visibility: (B, T, Nq)

pred_tracks = pred_tracks[0].cpu().numpy()          # (T, Nq, 2)
pred_visibility = pred_visibility[0].cpu().numpy()  # (T, Nq)

# apply visibility threshold and kill tracks when they go invisible
vis_mask = pred_visibility > VIS_THRESH           # (T, Nq) bool
# set invisible positions to NaN
pred_tracks[~vis_mask] = np.nan

# inject new SuperPoint points over time for new objects
# every NEW_POINT_INTERVAL frames
# only points far from existing tracks at that frame
all_extra_tracks = []
all_extra_vis = []

for frame_idx in range(0, T, NEW_POINT_INTERVAL):
    frame_kpts = kpts_all[frame_idx]  # (M, 2) in original resolution

    if frame_kpts.size == 0:
        continue

    # scale to resized coords
    frame_kpts_resized = np.zeros_like(frame_kpts, dtype=np.float32)
    frame_kpts_resized[:, 0] = frame_kpts[:, 0] * scale_x
    frame_kpts_resized[:, 1] = frame_kpts[:, 1] * scale_y

    # existing tracks at this frame
    existing_xy = pred_tracks[frame_idx]  # (Nq_total_so_far?, 2)

    # we only compare against valid existing points (not NaNs)
    valid_mask = ~np.isnan(existing_xy[:, 0])
    existing_xy_valid = existing_xy[valid_mask]

    new_queries_list = []
    for (xr, yr) in frame_kpts_resized:
        if existing_xy_valid.shape[0] > 0:
            dists = np.linalg.norm(existing_xy_valid - np.array([xr, yr]), axis=1)
            if np.nanmin(dists) <= DIST_THRESH:
                continue  # too close to an existing track, skip
        new_queries_list.append([float(frame_idx), xr, yr])

        if len(new_queries_list) >= MAX_NEW_POINTS_PER_INJECTION:
            break

    if len(new_queries_list) == 0:
        continue

    new_queries = torch.tensor(new_queries_list, dtype=torch.float32)[None]  # (1, K, 3)
    if torch.cuda.is_available():
        new_queries = new_queries.cuda()

    with torch.no_grad():
        new_tr, new_vis = model(
            video,
            queries=new_queries,
            backward_tracking=False   # forward-only for newly arriving points
        )

    new_tr = new_tr[0].cpu().numpy()   # (T, K, 2)
    new_vis = new_vis[0].cpu().numpy() # (T, K)

    # apply same visibility threshold + NaN masking
    new_vis_mask = new_vis > VIS_THRESH
    new_tr[~new_vis_mask] = np.nan

    all_extra_tracks.append(new_tr)
    all_extra_vis.append(new_vis)

# if we added any new tracks, concatenate them onto the main arrays
if len(all_extra_tracks) > 0:
    extra_tracks = np.concatenate(all_extra_tracks, axis=1)  # (T, sum_K, 2)
    extra_vis = np.concatenate(all_extra_vis, axis=1)        # (T, sum_K)

    pred_tracks = np.concatenate([pred_tracks, extra_tracks], axis=1)
    pred_visibility = np.concatenate([pred_visibility, extra_vis], axis=1)

    print("Added extra tracks for new objects. New shape:", pred_tracks.shape)

# save raw tracks + metadata to NPZ for later pose/depth
out_path = os.path.join(SAVE_DIR, "cotracker_superpoint_tracks.npz")
np.savez(
    out_path,
    tracks=pred_tracks,              # (T, N_total, 2) in resized coords
    visibility=pred_visibility,      # (T, N_total)
    query_frame=QUERY_FRAME,
    resized_height=Hr,
    resized_width=Wr,
    original_height=H0,
    original_width=W0,
    scale_x=scale_x,
    scale_y=scale_y,
    vis_thresh=VIS_THRESH,
    new_point_interval=NEW_POINT_INTERVAL,
)

print("Saved tracks to:", out_path)

# copy arrays
tracks_clean = pred_tracks.copy()
vis_clean    = pred_visibility.copy()

# mark invalid points as invisible
invalid = (
    np.isnan(tracks_clean[..., 0]) |
    np.isnan(tracks_clean[..., 1])
)
vis_clean[invalid] = 0

# forward-fill NaN values with last valid position
for track_idx in range(tracks_clean.shape[1]):
    track = tracks_clean[:, track_idx, :]  # (T, 2)
    mask = ~np.isnan(track[:, 0])
    
    if mask.any():
        # Forward fill
        last_valid = None
        for t in range(len(track)):
            if mask[t]:
                last_valid = track[t].copy()
            elif last_valid is not None:
                track[t] = last_valid

# after forward-fill, any remaining NaN can be set to (0, 0) safely
tracks_clean = np.nan_to_num(tracks_clean, nan=0.0)

# 5. Clip to image bounds
tracks_clean[..., 0] = np.clip(tracks_clean[..., 0], 0, Wr - 1)
tracks_clean[..., 1] = np.clip(tracks_clean[..., 1], 0, Hr - 1)

# da viz with tracks
vis_trails = Visualizer(
    save_dir=SAVE_DIR,
    linewidth=2,
    mode='cool',
    tracks_leave_trace=-1,  # -1 means full history
)

vis_trails.visualize(
    video=video,
    tracks=torch.from_numpy(tracks_clean)[None],     # (1, T, N, 2)
    visibility=torch.from_numpy(vis_clean)[None],    # (1, T, N)
    filename="superpoint_with_trails"
)

print("Track history visualization saved to:", SAVE_DIR)

# viz of only floating points
vis_points = Visualizer(
    save_dir=SAVE_DIR,
    linewidth=2,
    mode='cool',
    tracks_leave_trace=0,  # 0 means no history, just current points
)

vis_points.visualize(
    video=video,
    tracks=torch.from_numpy(tracks_clean)[None],     # (1, T, N, 2)
    visibility=torch.from_numpy(vis_clean)[None],    # (1, T, N)
    filename="superpoint_floating_points"
)

print("Points-only visualization saved to:", SAVE_DIR)
print("All visualizations complete!")