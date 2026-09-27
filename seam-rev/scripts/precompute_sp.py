#!/usr/bin/env python
"""
Precompute superpixel labels for VOC2012 (optional, for offline caching).

Note: SEAM uses RandomCrop(448) augmentation, so offline precomputation is
not directly compatible unless you precompute at a fixed resolution and
apply the same crop to superpixel labels. For MVP, use online generation instead.

Usage:
    python scripts/precompute_sp.py --voc12_root VOC2012 --split train
"""

import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc12_root', default='VOC2012', type=str)
    parser.add_argument('--split', default='val', type=str,
                        choices=['train', 'val', 'train_aug'])
    parser.add_argument('--out_dir', default='./precomputed_sp', type=str)
    parser.add_argument('--region_size', default=20, type=int)
    parser.add_argument('--ruler', default=10.0, type=float)
    args = parser.parse_args()

    # Read image list
    list_file = os.path.join('voc12', f'{args.split}.txt')
    if not os.path.exists(list_file):
        print(f'List file not found: {list_file}')
        return

    with open(list_file) as f:
        lines = f.read().strip().split('\n')
    image_ids = [line.split(' ')[0][-15:-4] if ' ' in line
                 else line.strip()[:11]
                 for line in lines]

    img_dir = os.path.join(args.voc12_root, 'JPEGImages')
    out_dir = os.path.join(args.out_dir, args.split)
    os.makedirs(out_dir, exist_ok=True)

    print(f'Precomputing superpixels for {len(image_ids)} images...')
    for img_id in tqdm(image_ids):
        out_path = os.path.join(out_dir, f'{img_id}.npy')
        if os.path.exists(out_path):
            continue

        img_path = os.path.join(img_dir, f'{img_id}.jpg')
        img = cv2.imread(img_path)
        if img is None:
            print(f'  Cannot read: {img_path}')
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        slic = cv2.ximgproc.createSuperpixelSLIC(
            img_rgb,
            algorithm=cv2.ximgproc.SLIC,
            region_size=args.region_size,
            ruler=args.ruler,
        )
        slic.iterate(10)
        slic.enforceLabelConnectivity()
        labels = slic.getLabels()

        # Remap to contiguous IDs
        unique = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique)}
        labels_remapped = np.vectorize(label_map.get)(labels).astype(np.int32)

        np.save(out_path, labels_remapped)

    print(f'Done. Saved to {out_dir}')


if __name__ == '__main__':
    main()
