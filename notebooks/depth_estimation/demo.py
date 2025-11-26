import os
import sys
import numpy as np
import torch
import cv2

# ------------------------------------------------------------
# Add VDA repo to Python path
# ------------------------------------------------------------
VDA_ROOT = os.path.expanduser("~/vda")
sys.path.append(os.path.join(VDA_ROOT))

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import save_video   # we KEEP save_video from repo


# ------------------------------------------------------------
# User config
# ------------------------------------------------------------
VIDEO_PATH = os.path.expanduser("../data/videos/IMG_0171.MOV")
OUTPUT_DIR = os.path.expanduser("../data/depth/IMG_0171_depth")
CHECKPOINT = os.path.expanduser("~/vda/checkpoints/video_depth_anything_vitl.pth")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Robust frame loader to replace buggy read_video_frames
# ------------------------------------------------------------
def load_video_frames(path, max_res=1280):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-2:
        fps = 30.0  # fallback

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        TARGET_H, TARGET_W = 720, 960
        frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    cap.release()
    return np.stack(frames, axis=0).astype(np.uint8), fps


# ------------------------------------------------------------
# Run full demo: VIS mp4 + SRC mp4 + optional NPZ
# ------------------------------------------------------------
def run_vda(video_path, checkpoint_path, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = VideoDepthAnything(
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
        metric=False
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    # Load frames safely
    frames, fps = load_video_frames(video_path, max_res=1280)

    # Save SRC video, just like demo
    video_name = os.path.basename(video_path)
    # src_path = os.path.join(out_dir, os.path.splitext(video_name)[0] + "_src.mp4")
    # save_video(frames, src_path, fps=fps)
    # print("Saved:", src_path)

    # Predict depth
    depths, _ = model.infer_video_depth(
        frames,
        fps,
        input_size=518,
        device=device,
        fp32=False
    )

    # Save DEPTH VIS video
    depth_vis_path = os.path.join(out_dir, os.path.splitext(video_name)[0] + "_vis.mp4")
    save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=False)
    print("Saved depth viz:", depth_vis_path)

    # Save raw depth to NPZ
    npz_path = os.path.join(out_dir, os.path.splitext(video_name)[0] + "_depths.npz")
    np.savez_compressed(npz_path, depths=depths)
    print("Saved depth npz:", npz_path)


# ------------------------------------------------------------
if __name__ == "__main__":
    run_vda(VIDEO_PATH, CHECKPOINT, OUTPUT_DIR)