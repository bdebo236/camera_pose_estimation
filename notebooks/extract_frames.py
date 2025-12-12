import cv2
import os

VIDEO_NAME = "IMG_0309"
VIDEO = f"/home/debbanerjee/camera_pose_estimation/data/videos/{VIDEO_NAME}.MOV"
OUTDIR = f"/home/debbanerjee/camera_pose_estimation/data/frames/{VIDEO_NAME}_frames"

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