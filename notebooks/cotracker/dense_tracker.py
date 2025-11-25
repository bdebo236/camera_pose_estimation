import os
import sys
import numpy as np
import torch
import cv2

# CoTracker repo
sys.path.append(os.path.expanduser("~/co-tracker"))

from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
VIDEO_PATH      = "../data/videos/IMG_0171.MOV"
SP_KPTS_PATH    = "../data/tracks/IMG_0171_tracks/superpoint_kpts.npz"
CHECKPOINT      = os.path.expanduser("~/co-tracker/checkpoints/scaled_offline.pth")
SAVE_DIR        = "../data/tracks/IMG_0171_cotracker_superpoint"
QUERY_FRAME     = 20        # like grid_query_frame=20
MAX_POINTS      = 500       # limit for stability / speed

# Original resolution (where SuperPoint was run)
H0, W0 = 1440, 1920
# Downsampled resolution used for CoTracker demo
Hr, Wr = 720, 960

os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Load video and resize (same logic as fixed demo)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# 2. Load SuperPoint keypoints and pick those from QUERY_FRAME
# ----------------------------------------------------------------------
sp = np.load(SP_KPTS_PATH, allow_pickle=True)
kpts_all = sp["keypoints"] # list length T, each (N_i, 2)

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

# ----------------------------------------------------------------------
# 3. Build CoTracker queries tensor
#    Format: [time, x, y] per point
# ----------------------------------------------------------------------
Nq = kpts_q_resized.shape[0]
queries_np = np.zeros((Nq, 3), dtype=np.float32)
queries_np[:, 0] = float(QUERY_FRAME)           # time index
queries_np[:, 1:] = kpts_q_resized              # x, y

queries = torch.from_numpy(queries_np)          # (Nq, 3)
if torch.cuda.is_available():
    queries = queries.cuda()

# add batch dim for model
queries = queries[None]  # (1, Nq, 3)

# ----------------------------------------------------------------------
# 4. Run CoTracker with backward=True (like grid_query_20_backward)
# ----------------------------------------------------------------------
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

pred_tracks = pred_tracks[0].cpu().numpy()       # (T, Nq, 2)
pred_visibility = pred_visibility[0].cpu().numpy()  # (T, Nq)

# ----------------------------------------------------------------------
# 5. Save raw tracks + metadata to NPZ for later pose/depth
# ----------------------------------------------------------------------
out_path = os.path.join(SAVE_DIR, "cotracker_superpoint_tracks.npz")
np.savez(
    out_path,
    tracks=pred_tracks,              # (T, Nq, 2) in resized coords
    visibility=pred_visibility,      # (T, Nq)
    query_frame=QUERY_FRAME,
    resized_height=Hr,
    resized_width=Wr,
    original_height=H0,
    original_width=W0,
    scale_x=scale_x,
    scale_y=scale_y,
)

print("Saved tracks to:", out_path)

# ----------------------------------------------------------------------
# 6. Also visualize like demo (optional)
# ----------------------------------------------------------------------
vis = Visualizer(
    save_dir=SAVE_DIR,
    linewidth=2,
    mode='cool',
    tracks_leave_trace=-1,
)

vis.visualize(
    video=video,
    tracks=torch.from_numpy(pred_tracks)[None],          # back to (1, T, Nq, 2)
    visibility=torch.from_numpy(pred_visibility)[None],  # back to (1, T, Nq)
    filename="superpoint_queries_backward"
)

print("Visualization saved to:", SAVE_DIR)