import os
import sys
import torch
import numpy as np
import cv2

# -----------------------------------------------------
# Paths
# -----------------------------------------------------
VIDEO_PATH = "../data/videos/IMG_0171.MOV"
SAVE_PATH  = "../data/tracks/IMG_0171_tracks/superpoint_kpts.npz"

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# -----------------------------------------------------
# Import SuperPoint (from your SuperGlue folder)
# -----------------------------------------------------
sys.path.append(os.path.expanduser("~/superglue"))   # adjust if needed

from models.superpoint import SuperPoint

# -----------------------------------------------------
# Load video
# -----------------------------------------------------
def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame_gray)
    cap.release()
    return np.stack(frames, axis=0)  # (T, H, W)

video = load_video(VIDEO_PATH)  # (T, H, W)
T, H, W = video.shape
print("Loaded video:", video.shape)

# -----------------------------------------------------
# Initialize SuperPoint
# -----------------------------------------------------
sp_config = {
    'descriptor_dim': 256,
    'nms_radius': 3,
    'keypoint_threshold': 0.005,
    'max_keypoints': 2000,
}
superpoint = SuperPoint(sp_config)
superpoint = superpoint.cuda().eval()

# -----------------------------------------------------
# Extract keypoints for each frame
# -----------------------------------------------------
all_kpts = []      # list length T, each (N_i, 2)
all_scores = []    # (N_i,)
all_desc = []      # (256, N_i)

for t in range(T):
    frame = video[t]  # (H, W) uint8
    frame_norm = (frame / 255.).astype(np.float32)

    inp = torch.from_numpy(frame_norm)[None, None].cuda()  # (1,1,H,W)

    with torch.no_grad():
        out = superpoint({'image': inp})

    kpts = out['keypoints'][0].cpu().numpy()     # (N, 2)
    scores = out['scores'][0].cpu().numpy()      # (N,)
    desc = out['descriptors'][0].cpu().numpy().T # (N, 256) or transpose if needed

    all_kpts.append(kpts)
    all_scores.append(scores)
    all_desc.append(desc)

    print(f"Frame {t}: {kpts.shape[0]} keypoints")

# -----------------------------------------------------
# Save everything to NPZ
# -----------------------------------------------------
np.savez(
    SAVE_PATH,
    keypoints=np.array(all_kpts, dtype=object),
    scores=np.array(all_scores, dtype=object),
    descriptors=np.array(all_desc, dtype=object),
    height=H,
    width=W,
)

print("Saved SuperPoint keypoints to:", SAVE_PATH)