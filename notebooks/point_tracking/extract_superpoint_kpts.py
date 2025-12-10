import os
import sys
import torch
import numpy as np
import cv2

IMG_NAME  = "IMG_0171"
VIDEO_PATH = f"../data/videos/{IMG_NAME}.MOV"
SAVE_PATH  = f"../data/tracks/{IMG_NAME}_tracks/superpoint_kpts.npz"

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
sys.path.append(os.path.expanduser("~/superglue"))   # adjust if needed

from models.superpoint import SuperPoint

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

# initialize superPoint
sp_config = {
    'descriptor_dim': 256,
    'nms_radius': 3,
    'keypoint_threshold': 0.005,
    'max_keypoints': 2000,
}
superpoint = SuperPoint(sp_config)
superpoint = superpoint.cuda().eval()

# extract keypoints for each frame
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

# save
np.savez(
    SAVE_PATH,
    keypoints=np.array(all_kpts, dtype=object),
    scores=np.array(all_scores, dtype=object),
    descriptors=np.array(all_desc, dtype=object),
    height=H,
    width=W,
)

print("Saved SuperPoint keypoints to:", SAVE_PATH)