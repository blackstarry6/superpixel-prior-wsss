"""
SLIC semantic purity analysis (quantifies the confound for SPFR's prototype
pooling: how often do SLIC superpixels cross semantic boundaries?).

Uses the EXACT SLIC protocol of spfr/superpixel.py (OpenCV SLIC,
region_size=20, ruler=10.0, n_iter=5, fast 256px mode) applied to full
val images, then measures per-superpixel class purity against GT.

Outputs: analysis/slic_purity_analysis.txt

Usage: python analyze_slic_purity.py [--limit N]   (N=0 -> all 1449 images)
"""
import argparse
import numpy as np
import cv2
import os
from PIL import Image
from tqdm import tqdm

GT = 'VOC2012/SegmentationClass'
JPEG = 'VOC2012/JPEGImages'

REGION_SIZE = 20
RULER = 10.0
N_ITER = 5
PURITY_THRESHOLD = 0.8     # superpixel is "crossing" if purity below this
MIN_VALID = 10             # skip superpixels with fewer valid GT pixels


def slic_labels(img, region_size=REGION_SIZE, ruler=RULER, n_iter=N_ITER, fast=True):
    """Mirror of spfr/superpixel.py generate_superpixels (numpy-only)."""
    H, W = img.shape[:2]
    if fast and max(H, W) > 256:
        scale = 256.0 / max(H, W)
        H_s = int(round(H * scale)); W_s = int(round(W * scale))
        img_s = cv2.resize(img, (W_s, H_s), interpolation=cv2.INTER_LINEAR)
        rs = max(5, int(round(region_size * scale)))
    else:
        img_s = img; H_s, W_s = H, W; rs = region_size

    slic = cv2.ximgproc.createSuperpixelSLIC(
        img_s, algorithm=cv2.ximgproc.SLIC, region_size=rs, ruler=ruler)
    slic.iterate(n_iter)
    slic.enforceLabelConnectivity()
    lab = slic.getLabels()

    if (H_s, W_s) != (H, W):
        lab = cv2.resize(lab.astype(np.float32), (W, H),
                         interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='max images (0=all)')
    args = ap.parse_args()

    names = sorted(f[:-4] for f in os.listdir(GT) if f.endswith('.png'))
    # restrict to val split if available
    val_file = 'voc12/val.txt'
    if os.path.exists(val_file):
        val_names = [l.split(' ')[0].strip().split('/')[-1][:-4]
                     for l in open(val_file).read().strip().split('\n')]
        val_names = [n for n in val_names if n in set(names)]
        if len(val_names) == 1449:
            names = sorted(val_names)
    if args.limit:
        names = names[:args.limit]
    print(f'{len(names)} images, SLIC protocol: region_size={REGION_SIZE}, ruler={RULER}, n_iter={N_ITER}, fast=256px')

    os.makedirs('analysis', exist_ok=True)
    out_lines = []
    def out(s=''):
        print(s)
        out_lines.append(s)

    # aggregates
    total_sp = 0
    skipped_small = 0
    weighted_purity_sum = 0.0
    weighted_pixels = 0
    crossing_sp = 0
    crossing_pixels = 0
    two_class_sp = 0            # >=2 classes with share >=10%
    purities = []

    buckets = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]
    bucket_sp = np.zeros(len(buckets))

    for name in tqdm(names):
        img = np.array(Image.open(os.path.join(JPEG, name + '.jpg')).convert('RGB'))
        gt = np.array(Image.open(os.path.join(GT, name + '.png')))
        lab = slic_labels(img)
        valid = gt < 255

        for sp in np.unique(lab):
            m = lab == sp
            v = m & valid
            n = int(v.sum())
            if n < MIN_VALID:
                skipped_small += 1
                continue
            total_sp += 1
            cnt = np.bincount(gt[v], minlength=21)
            purity = float(cnt.max()) / n
            purities.append(purity)
            weighted_purity_sum += purity * n
            weighted_pixels += n
            if purity < PURITY_THRESHOLD:
                crossing_sp += 1
                crossing_pixels += n
            n_share = int((cnt / n >= 0.10).sum())
            if n_share >= 2:
                two_class_sp += 1
            for bi, (lo, hi) in enumerate(buckets):
                if lo <= purity < hi:
                    bucket_sp[bi] += 1
                    break

    purities = np.array(purities)
    out('=' * 70)
    out('  SLIC semantic purity analysis (training protocol, full val images)')
    out('=' * 70)
    out(f'  images: {len(names)}, superpixels: {total_sp}, skipped (valid<{MIN_VALID}): {skipped_small}')
    out(f'  mean purity (pixel-weighted): {weighted_purity_sum / weighted_pixels:.3f}')
    out(f'  median purity (per-superpixel): {np.median(purities):.3f}')
    out(f'  superpixels crossing semantic boundary (purity<{PURITY_THRESHOLD:.1f}): '
        f'{crossing_sp} / {total_sp} = {100.0 * crossing_sp / total_sp:.1f}%')
    out(f'  valid pixels inside crossing superpixels: {100.0 * crossing_pixels / weighted_pixels:.1f}%')
    out(f'  superpixels containing >=2 classes (share>=10%): '
        f'{two_class_sp} / {total_sp} = {100.0 * two_class_sp / total_sp:.1f}%')
    out('')
    out('  purity histogram (share of superpixels):')
    for (lo, hi), c in zip(buckets, bucket_sp):
        out(f'    [{lo:.2f}, {hi:.2f}): {100.0 * c / total_sp:5.1f}%')

    with open('analysis/slic_purity_analysis.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines) + '\n')
    print('\nSaved: analysis/slic_purity_analysis.txt')


if __name__ == '__main__':
    main()
