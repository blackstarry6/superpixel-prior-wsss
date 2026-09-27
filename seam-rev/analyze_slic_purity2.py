"""
SLIC purity measurement matrix (round-4 review fix).

Measures superpixel semantic purity for BOTH experiment protocols on BOTH
image sets (2x2 matrix), with per-protocol superpixel counts:

  region_size=20  -> Experiment 1 protocol (classification stage)
  region_size=25  -> Experiment 2 protocol (segmentation stage, on 448 crops;
                     full-image measurement approximates the same granularity
                     because fast mode scales region_size with image size)
  splits: val (1449, GT: SegmentationClass) and train_aug (10582, GT: SegmentationClassAug)

Purity definitions replicate analyze_slic_purity.py exactly (purity = majority
class share among valid GT pixels; crossing if purity < 0.8; skip valid<10;
same histogram buckets). The val+rs20 cell must reproduce the original
numbers (607069 superpixels, mean 0.987, 2.9% crossing, 93.9% >=0.95) as a
built-in correctness check. The per-superpixel python loop is replaced by a
single vectorized bincount pass.

Usage:
  python analyze_slic_purity2.py --split val --region_size 20
  python analyze_slic_purity2.py --split train_aug --region_size 25
"""
import argparse
import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

VOC_ROOT = '../VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
JPEG = os.path.join(VOC_ROOT, 'JPEGImages')
GT_VAL = os.path.join(VOC_ROOT, 'SegmentationClass')
GT_AUG = os.path.join(VOC_ROOT, 'SegmentationClassAug')

RULER = 10.0
N_ITER = 5
PURITY_THRESHOLD = 0.8
MIN_VALID = 10
BUCKETS = np.array([0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.01])


def slic_labels(img, region_size, ruler=RULER, n_iter=N_ITER, fast=True):
    """Identical to analyze_slic_purity.py / spfr.superpixel protocol."""
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


def load_state(ckpt):
    if os.path.exists(ckpt):
        d = np.load(ckpt, allow_pickle=True)
        return (list(d['done']), d['purities'], int(d['total_sp']), int(d['skipped']),
                int(d['crossing_sp']), int(d['crossing_pix']), int(d['two_class_sp']),
                float(d['wp_sum']), int(d['wpix']), d['bucket_sp'], int(d['raw_sp'])
                ), d['names']
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['val', 'train_aug'], required=True)
    ap.add_argument('--region_size', type=int, required=True)
    ap.add_argument('--limit', type=int, default=0, help='max images (0=all)')
    args = ap.parse_args()

    list_file = os.path.join('voc12', f'{args.split}.txt')
    names_all = [l.split(' ')[0].strip().split('/')[-1][:-4]
                 for l in open(list_file).read().strip().split('\n') if l.strip()]
    gt_dir = GT_VAL if args.split == 'val' else GT_AUG
    names = [n for n in names_all
             if os.path.exists(os.path.join(gt_dir, n + '.png'))
             and os.path.exists(os.path.join(JPEG, n + '.jpg'))]
    if args.limit:
        names = names[:args.limit]
    print(f'{args.split}: {len(names)}/{len(names_all)} images, region_size={args.region_size}, '
          f'ruler={RULER}, n_iter={N_ITER}, fast=256px, GT={gt_dir}')

    os.makedirs('analysis', exist_ok=True)
    tag = f'{args.split}_rs{args.region_size}'
    ckpt = f'analysis/_ckpt_{tag}.npz'
    st, _ = load_state(ckpt)
    if st is None:
        done, purities = set(), np.zeros(0, dtype=np.float32)
        total_sp = skipped = crossing_sp = crossing_pix = two_class_sp = raw_sp = wpix = 0
        wp_sum = 0.0
        bucket_sp = np.zeros(len(BUCKETS) - 1)
    else:
        (done, purities, total_sp, skipped, crossing_sp, crossing_pix, two_class_sp,
         wp_sum, wpix, bucket_sp, raw_sp) = st
        print(f'resume: {len(done)} images already done')

    todo = [n for n in names if n not in done]
    new_p = []
    for name in tqdm(todo, desc=tag, initial=len(done), total=len(names)):
        img = np.array(Image.open(os.path.join(JPEG, name + '.jpg')).convert('RGB'))
        gt = np.array(Image.open(os.path.join(gt_dir, name + '.png')))
        lab = slic_labels(img, args.region_size)
        valid = gt < 255
        K = int(lab.max()) + 1
        lab_v = lab[valid].astype(np.int64)
        gt_v = gt[valid].astype(np.int64)
        assert gt_v.max() < 21, name
        comb = np.bincount(lab_v * 21 + gt_v, minlength=K * 21).reshape(K, 21)
        n = comb.sum(1)
        raw_sp += K
        keep = n >= MIN_VALID
        skipped += int((~keep).sum())
        total_sp += int(keep.sum())
        p = (comb[np.arange(K), comb.argmax(1)] / np.maximum(n, 1)).astype(np.float32)
        pk = p[keep]
        nk = n[keep]
        new_p.append(pk)
        wp_sum += float((pk * nk).sum()); wpix += int(nk.sum())
        cr = pk < PURITY_THRESHOLD
        crossing_sp += int(cr.sum()); crossing_pix += int(nk[cr].sum())
        shares = comb[keep] / nk[:, None]
        two_class_sp += int(((shares >= 0.10).sum(1) >= 2).sum())
        bucket_sp += np.histogram(pk, bins=BUCKETS)[0]
        done.add(name)
        if len(done) % 1000 == 0:
            purities = np.concatenate([purities] + new_p) if new_p else purities
            new_p = []
            np.savez(ckpt, done=np.array(sorted(done)), purities=purities,
                     total_sp=total_sp, skipped=skipped, crossing_sp=crossing_sp,
                     crossing_pix=crossing_pix, two_class_sp=two_class_sp,
                     wp_sum=wp_sum, wpix=wpix, bucket_sp=bucket_sp, raw_sp=raw_sp,
                     names=np.array(names))

    purities = np.concatenate([purities] + new_p) if new_p else purities
    if os.path.exists(ckpt):
        os.remove(ckpt)

    lines = []
    lines.append('=' * 70)
    lines.append(f'  SLIC purity | split={args.split} | region_size={args.region_size} '
                 f'(ruler={RULER}, n_iter={N_ITER}, fast=256px)')
    lines.append('=' * 70)
    lines.append(f'  images: {len(done)}, superpixels (kept): {total_sp}, '
                 f'skipped (valid<{MIN_VALID}): {skipped}')
    lines.append(f'  superpixels per image: kept {total_sp / max(len(done),1):.1f}, '
                 f'raw {raw_sp / max(len(done),1):.1f}')
    lines.append(f'  mean purity (pixel-weighted): {wp_sum / max(wpix,1):.3f}')
    lines.append(f'  median purity (per-superpixel): {np.median(purities):.3f}')
    lines.append(f'  crossing semantic boundary (purity<{PURITY_THRESHOLD:.1f}): '
                 f'{crossing_sp} / {total_sp} = {100.0 * crossing_sp / max(total_sp,1):.1f}%')
    lines.append(f'  valid pixels inside crossing superpixels: {100.0 * crossing_pix / max(wpix,1):.1f}%')
    lines.append(f'  >=2 classes (share>=10%): {two_class_sp} / {total_sp} = '
                 f'{100.0 * two_class_sp / max(total_sp,1):.1f}%')
    lines.append('')
    lines.append('  purity histogram (share of superpixels):')
    for i in range(len(BUCKETS) - 1):
        lines.append(f'    [{BUCKETS[i]:.2f}, {BUCKETS[i+1]:.2f}): '
                     f'{100.0 * bucket_sp[i] / max(total_sp,1):5.1f}%')
    out_path = f'analysis/slic_purity_matrix_{tag}.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
