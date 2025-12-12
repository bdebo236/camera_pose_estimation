# Camera Pose Estimation

This repository contains code for estimating camera pose from monocular video.

---

## Repository Structure

```text
camera_pose_estimation/
├── notebooks/
│   ├── core/                  # Core pose estimation modules
│   │   ├── loader.py           # Video loading and camera intrinsics
│   │   ├── correspondences.py  # 2D–2D and 3D–2D correspondences
│   │   ├── pose_estimation.py  # Essential matrix, PnP, RANSAC solvers
│   │   ├── trajectory.py       # Global trajectory integration
│   │   ├── ba.py               # (Optional) bundle adjustment
│   │   └── viz.py              # Visualization utilities
│   │
│   ├── main.py                 # Main depth-assisted pose pipeline
│   ├── main_2d.py              # 2D-only pose estimation (no depth)
│   └── main_with_ba.py         # Pipeline with bundle adjustment (experimented, unused)
│
├── data/
│   ├── videos/                 # Input video files
│   ├── frames/                 # Extracted video frames
│   ├── depth/                  # Predicted depth maps
│   └── tracks/                 # Feature tracking outputs
│
├── results/                    # Estimated trajectories, plots, metrics
├── requirements.txt
└── README.md
```

---

## Running the Code

Place your input video in:

```text
data/videos/
```

Open a main script and set:

```python
VIDEO_NAME = "IMG_0171"
```

Run the depth-assisted pipeline:

```bash
python notebooks/main.py
```
