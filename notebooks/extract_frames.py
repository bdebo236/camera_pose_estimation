import cv2
import os

VIDEO = "/home/debbanerjee/camera_pose_estimation/data/videos/IMG_0112.MOV"
OUTDIR = "/home/debbanerjee/camera_pose_estimation/data/frames/IMG_0112_frames"

os.makedirs(OUTDIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)
i = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    fname = os.path.join(OUTDIR, f"{i:06d}.jpg")
    cv2.imwrite(fname, frame)
    i += 1

cap.release()
print("Done. Extracted", i, "frames.")