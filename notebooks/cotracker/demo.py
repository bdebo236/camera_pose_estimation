import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# YOUR HPC SETUP: CoTracker lives OUTSIDE main repo
# ---------------------------------------------------------
import sys
sys.path.append(os.path.expanduser("~/co-tracker"))

from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

# ---------------------------------------------------------
# INPUTS (your video + your cluster path)
# ---------------------------------------------------------
VIDEO_PATH = "../data/videos/IMG_0171.MOV"
CHECKPOINT = os.path.expanduser("~/co-tracker/checkpoints/scaled_offline.pth")
SAVE_DIR   = "../data/tracks/IMG_0171_demo"

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------
# LOAD VIDEO (same logic)
# ---------------------------------------------------------
video_np = read_video_from_path(VIDEO_PATH)   # T, H, W, C
video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float()   # (1, T, C, H, W)

print("Before resize:", video.shape)   # (1, 350, 3, 1920, 1440)

# reshape -> resize -> reshape back
B, T, C, H, W = video.shape
video = video.reshape(B*T, C, H, W)    # (350, 3, 1920, 1440)
video = torch.nn.functional.interpolate(video, size=(720, 960), mode="bilinear")
video = video.reshape(B, T, C, 720, 960)

print("After resize:", video.shape)    # (1, 350, 3, 720, 960)

if torch.cuda.is_available():
    video = video.cuda()

# ---------------------------------------------------------
# LOAD MODEL (same logic)
# ---------------------------------------------------------
model = CoTrackerPredictor(checkpoint=CHECKPOINT)
if torch.cuda.is_available():
    model = model.cuda()

# ---------------------------------------------------------
# 1. GRID TRACKING (same as demo)
# ---------------------------------------------------------
pred_tracks, pred_visibility = model(video, grid_size=30)

vis = Visualizer(save_dir=SAVE_DIR, pad_value=100)
vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, filename="grid_demo")

# ---------------------------------------------------------
# 2. MANUAL QUERIES (same logic)
# ---------------------------------------------------------
queries = torch.tensor([
    [0., 400., 350.],
    [10., 600., 500.],
    [20., 750., 600.],
    [30., 900., 200.]
])
if torch.cuda.is_available():
    queries = queries.cuda()

pred_tracks, pred_visibility = model(video, queries=queries[None])

vis = Visualizer(save_dir=SAVE_DIR, linewidth=6, mode='cool', tracks_leave_trace=-1)
vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, filename='queries')

# ---------------------------------------------------------
# 3. MANUAL QUERIES WITH BACKWARD TRACKING (same logic)
# ---------------------------------------------------------
pred_tracks, pred_visibility = model(video, queries=queries[None], backward_tracking=True)
vis.visualize(video, pred_tracks, pred_visibility, filename="queries_backward")

# ---------------------------------------------------------
# 4. GRID STARTING FROM SOME MIDDLE FRAME (same logic)
# ---------------------------------------------------------
grid_size = 30
grid_query_frame = 20

pred_tracks, pred_visibility = model(
    video, grid_size=grid_size, grid_query_frame=grid_query_frame
)

vis = Visualizer(save_dir=SAVE_DIR, pad_value=100)
vis.visualize(
    video=video,
    tracks=pred_tracks,
    visibility=pred_visibility,
    filename='grid_query_20',
    query_frame=grid_query_frame
)

# with backward
pred_tracks, pred_visibility = model(
    video, grid_size=grid_size, grid_query_frame=grid_query_frame, backward_tracking=True
)

vis.visualize(
    video=video,
    tracks=pred_tracks,
    visibility=pred_visibility,
    filename='grid_query_20_backward'
)

# ---------------------------------------------------------
# 5. SEGMENTATION MASK EXAMPLE (same logic)
# ---------------------------------------------------------
from PIL import Image

MASK_PATH = "./assets/apple_mask.png"  # replace if needed
segm_mask = np.array(Image.open(MASK_PATH))

pred_tracks, pred_visibility = model(
    video,
    grid_size=100,
    segm_mask=torch.from_numpy(segm_mask)[None, None]
)

vis = Visualizer(save_dir=SAVE_DIR, pad_value=100, linewidth=2)
vis.visualize(
    video=video,
    tracks=pred_tracks,
    visibility=pred_visibility,
    filename='segm_grid'
)

# ---------------------------------------------------------
# 6. DENSE TRACKING (same logic)
# ---------------------------------------------------------
import torch.nn.functional as F

video_interp = F.interpolate(video[0], [200, 360], mode="bilinear")[None]

pred_tracks, pred_visibility = model(
    video_interp,
    grid_query_frame=20,
    backward_tracking=True
)

vis = Visualizer(save_dir=SAVE_DIR, pad_value=20, linewidth=1, mode='optical_flow')
vis.visualize(
    video=video_interp,
    tracks=pred_tracks,
    visibility=pred_visibility,
    filename='dense'
)

print("Done. All outputs saved to:", SAVE_DIR)