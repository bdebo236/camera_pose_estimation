import os
import sys
import numpy as np
import torch
import cv2

# Add VDA repo to Python path
VDA_ROOT = os.path.expanduser("~/vda")
sys.path.append(os.path.join(VDA_ROOT))

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import save_video  

# User config
# IMG_NAME  = "IMG_0112"
IMG_NAME  = "IMG_0309"
# IMG_NAME  = "IMG_0171"
VIDEO_PATH = os.path.expanduser(f"../data/videos/{IMG_NAME}.MOV")
OUTPUT_DIR = os.path.expanduser(f"../data/depth/{IMG_NAME}_depth")
CHECKPOINT = os.path.expanduser("~/vda/checkpoints/metric_video_depth_anything_vitl.pth")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Robust frame loader to replace buggy read_video_frames
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

# Add depth grid overlay to visualization frames
def add_depth_grid_overlay(vis_frames, depth_frames, grid_size=(4, 4)):
    frames_with_grid = vis_frames.copy()
    n_frames, h, w = depth_frames.shape
    rows, cols = grid_size
    
    for frame_idx in range(n_frames):
        frame = frames_with_grid[frame_idx]
        depth = depth_frames[frame_idx]
        
        # Calculate grid positions
        y_positions = np.linspace(h * 0.15, h * 0.85, rows).astype(int)
        x_positions = np.linspace(w * 0.15, w * 0.85, cols).astype(int)
        
        for y in y_positions:
            for x in x_positions:
                # Get depth value at this point
                depth_val = depth[y, x]
                
                # Draw crosshair
                cv2.drawMarker(frame, (x, y), (0, 255, 0), 
                              markerType=cv2.MARKER_CROSS, 
                              markerSize=15, thickness=2)
                
                # Format depth text
                depth_text = f"{depth_val:.2f}m"
                
                # Get text size for background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(
                    depth_text, font, font_scale, thickness
                )
                
                # Draw background rectangle
                text_x = x + 10
                text_y = y - 10
                cv2.rectangle(frame, 
                            (text_x - 2, text_y - text_h - 2),
                            (text_x + text_w + 2, text_y + baseline + 2),
                            (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(frame, depth_text, (text_x, text_y),
                           font, font_scale, (0, 255, 0), thickness)
    
    return frames_with_grid


# Run full demo: VIS mp4 + SRC mp4 + optional NPZ
def run_vda(video_path, checkpoint_path, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = VideoDepthAnything(
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
        metric=True
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    # Load frames safely
    frames, fps = load_video_frames(video_path, max_res=1280)

    # Predict depth
    depths, _ = model.infer_video_depth(
        frames,
        fps,
        input_size=518,
        device=device,
        fp32=False
    )

    # Save standard DEPTH VIS video (without grid)
    video_name = os.path.basename(video_path)
    depth_vis_path = os.path.join(out_dir, os.path.splitext(video_name)[0] + "_vis.mp4")
    save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=False)
    print("Saved depth viz:", depth_vis_path)

    # Create visualization frames with grid overlay
    # First, convert depths to vis frames the same way save_video does
    depth_min = depths.min()
    depth_max = depths.max()
    depths_normalized = (depths - depth_min) / (depth_max - depth_min)
    depths_normalized = (depths_normalized * 255).astype(np.uint8)
    
    # Apply colormap
    vis_frames = np.zeros((*depths_normalized.shape, 3), dtype=np.uint8)
    for i in range(len(depths_normalized)):
        vis_frames[i] = cv2.applyColorMap(depths_normalized[i], cv2.COLORMAP_INFERNO)
        vis_frames[i] = cv2.cvtColor(vis_frames[i], cv2.COLOR_BGR2RGB)
    
    # Add grid overlay
    vis_frames_with_grid = add_depth_grid_overlay(vis_frames, depths, grid_size=(4, 4))
    
    # Save video with grid
    depth_grid_path = os.path.join(out_dir, os.path.splitext(video_name)[0] + "_vis_grid.mp4")
    save_video(vis_frames_with_grid, depth_grid_path, fps=fps, is_depths=False, grayscale=False)
    print("Saved depth viz with grid:", depth_grid_path)

    # Save raw depth to NPZ
    npz_path = os.path.join(out_dir, os.path.splitext(video_name)[0] + "_depths.npz")
    np.savez_compressed(npz_path, depths=depths)
    print("Saved depth npz:", npz_path)

if __name__ == "__main__":
    run_vda(VIDEO_PATH, CHECKPOINT, OUTPUT_DIR)