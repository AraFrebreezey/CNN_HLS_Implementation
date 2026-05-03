import cv2
import os
import numpy as np

INPUT_IMAGES = [
    ("C:/Users/arafa/Xilinx_projects/Project/Project/test/Cyst.jpg",   "Cyst"),
    ("C:/Users/arafa/Xilinx_projects/Project/Project/test/Normal.jpg", "Normal"),
    ("C:/Users/arafa/Xilinx_projects/Project/Project/test/Stone.jpg",  "Stone"),
    ("C:/Users/arafa/Xilinx_projects/Project/Project/test/Tumor.jpg",  "Tumor"),
]

OUTPUT_DIR  = "C:/Users/arafa/Xilinx_projects/Project/Project/test/resized64"
IMAGE_SIZE  = (64, 64)


os.makedirs(OUTPUT_DIR, exist_ok=True)

for src_path, name in INPUT_IMAGES:
    img = cv2.imread(src_path)
    if img is None:
        print(f"[ERROR] Could not load: {src_path}")
        continue

    print(f"Processing {name}.jpg: original size = {img.shape[1]}x{img.shape[0]}")

    # Resize using INTER_LINEAR -- same as training preprocessing
    resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

    # Save as PNG (lossless -- no JPEG artefacts)
    out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    cv2.imwrite(out_path, resized)
    print(f"  Saved: {out_path}")




print(f"Done. Pre-resized images saved to: {OUTPUT_DIR}")

