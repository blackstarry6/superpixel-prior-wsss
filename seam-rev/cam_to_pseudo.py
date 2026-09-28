"""
Convert CAM .npy to pseudo label PNGs.
Usage: python cam_to_pseudo.py --cam_dir ./cam_train --out_dir ./pseudo_train
"""
import numpy as np
import cv2
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cam_dir', required=True)
parser.add_argument('--out_dir', required=True)
parser.add_argument('--img_dir', default='VOC2012/JPEGImages')
parser.add_argument('--threshold', type=float, default=0.25)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

cam_files = [f for f in os.listdir(args.cam_dir) if f.endswith('.npy')]
print(f'Converting {len(cam_files)} CAM files to pseudo labels...')

for cam_file in tqdm(cam_files):
    img_name = cam_file[:-4]
    out_path = os.path.join(args.out_dir, img_name + '.png')
    if os.path.exists(out_path):
        continue

    cam_dict = np.load(os.path.join(args.cam_dir, cam_file), allow_pickle=True).item()

    img_path = os.path.join(args.img_dir, img_name + '.jpg')
    img = cv2.imread(img_path)
    if img is None:
        continue
    H, W = img.shape[:2]

    # Build full CAM (21, H, W): [bg, cls_0, cls_1, ..., cls_19]
    cam_full = np.zeros((21, H, W), dtype=np.float32)
    for cls_idx, cam_cls in cam_dict.items():
        h_cam, w_cam = cam_cls.shape
        if h_cam != H or w_cam != W:
            cam_cls = cv2.resize(cam_cls, (W, H), interpolation=cv2.INTER_LINEAR)
        cam_full[cls_idx + 1] = cam_cls

    # Min-max normalize per foreground class
    for c in range(1, 21):
        cmax = cam_full[c].max()
        cmin = cam_full[c].min()
        if cmax > cmin:
            cam_full[c] = (cam_full[c] - cmin) / (cmax - cmin)

    # Background = 1 - max foreground
    fg_max = cam_full[1:].max(axis=0)
    cam_full[0] = 1.0 - fg_max

    # Assign labels
    pseudo = np.full((H, W), 255, dtype=np.uint8)
    fg_mask = fg_max > args.threshold
    bg_mask = cam_full[0] > fg_max
    pseudo[fg_mask] = cam_full[1:].argmax(axis=0)[fg_mask] + 1
    pseudo[bg_mask] = 0

    cv2.imwrite(out_path, pseudo)

print(f'Done. {len(cam_files)} pseudo labels saved to {args.out_dir}')
