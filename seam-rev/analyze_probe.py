"""
Feature probe analysis (offline, no training).

Probe 1 (boundary alignment): ratio of conv5 feature-gradient magnitude at
semantic-boundary cells vs interior cells, compared with the same ratio for
raw RGB. H1 prediction: raw features are smooth — ratio close to 1 even at
semantic boundaries.

Probe 2 (classification vs similarity structure): linear probe (ridge
regression on conv5 pixel features) vs nearest-class-prototype assignment
(cosine). H1 prediction: linear >> prototype — features encode class info
linearly but are not organized as a similarity metric.

Checkpoints: baseline (resnet38_SEAM.pth) and FT-only s1 (finetune_only_s1.pth).
Images: --n val images (default 300), resized to 448x448 (training size).

Outputs: analysis/probe_analysis.txt

Usage: python analyze_probe.py [--n 300]
"""
import argparse
import os
import sys
import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import RidgeClassifier

sys.path.insert(0, '.')
from network.resnet38_SEAM import Net

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
GT = 'VOC2012/SegmentationClass'
JPEG = 'VOC2012/JPEGImages'
SIZE = 448
CELL = SIZE // 28  # 16


def load_model(path):
    model = Net().cuda().eval()
    sd = torch.load(path, map_location='cpu')
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model


def cell_stats(gt, h, w):
    """Boundary fraction and valid fraction per (h,w) cell, plus majority label."""
    g = gt.astype(np.int64)
    valid = g < 255
    gv = np.where(valid, g, -1)
    bnd = np.zeros_like(g, dtype=bool)
    for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        gy = np.roll(gv, dy, axis=0)
        gx = np.roll(gv, dx, axis=1)
        bnd |= (gv != -1) & (gy != -1) & (gv != gy)
    ch, cw = SIZE // h, SIZE // w
    bf = bnd.reshape(h, ch, w, cw).mean(axis=(1, 3))
    vf = valid.reshape(h, ch, w, cw).mean(axis=(1, 3))
    bfrac = np.where(vf > 0.5, bf / np.maximum(vf, 1e-6), 0.0).astype(np.float32)
    # majority label per cell
    cnt = np.zeros((h, w, 21), np.int64)
    lbl = gv.reshape(h, ch, w, cw)
    for c in range(21):
        cnt[:, :, c] = (lbl == c).sum(axis=(1, 3))
    maj = cnt.argmax(axis=2)
    return bfrac, vf, maj


def feat_grad_mag(f):
    """Per-cell mean gradient magnitude of (1024,h,w) feature."""
    gx = np.linalg.norm(np.diff(f, axis=2), axis=0)   # (h, w-1)
    gy = np.linalg.norm(np.diff(f, axis=1), axis=0)   # (h-1, w)
    g = np.zeros((f.shape[1], f.shape[2]), np.float32)
    g[:, :-1] += gx
    g[:, 1:] += gx
    g[:-1, :] += gy
    g[1:, :] += gy
    return g / 4.0


def rgb_grad_mag(img):
    """Per-cell mean RGB gradient magnitude."""
    g = np.zeros((img.shape[0], img.shape[1]), np.float32)
    for c in range(3):
        ch = img[:, :, c].astype(np.float32)
        gx = np.abs(np.diff(ch, axis=1))
        gy = np.abs(np.diff(ch, axis=0))
        g[:, :-1] += gx
        g[:, 1:] += gx
        g[:-1, :] += gy
        g[1:, :] += gy
    return g / 12.0


def boundary_ratio(feat_grad, bfrac, vf):
    bmask = (vf >= 0.5) & (bfrac >= 0.05)
    imask = (vf >= 0.5) & (bfrac <= 0.0)
    if bmask.sum() < 20 or imask.sum() < 20:
        return np.nan, int(bmask.sum()), int(imask.sum())
    return float(feat_grad[bmask].mean() / max(feat_grad[imask].mean(), 1e-8)), int(bmask.sum()), int(imask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    args = ap.parse_args()

    val_names = [l.split(' ')[0].strip().split('/')[-1][:-4]
                 for l in open('voc12/val.txt').read().strip().split('\n')]
    names = val_names[:args.n]

    out_lines = []
    def out(s=''):
        print(s)
        out_lines.append(s)

    out('=' * 70)
    out('  Feature probe analysis (offline; no training)')
    out('=' * 70)
    out(f'  images: {len(names)} (val), input 448x448, conv5 (1024, H, W)')
    out('')

    for path, tag in [('../SEAM_model/resnet38_SEAM.pth', 'baseline'),
                      ('finetune_only_s1.pth', 'FT-only s1')]:
        model = load_model(path)
        b_ratio_f, b_ratio_rgb = [], []
        n_bcell, n_icell = 0, 0
        F_list, y_list = [], []
        for name in tqdm(names, desc=tag):
            img = np.array(Image.open(os.path.join(JPEG, name + '.jpg')).convert('RGB'))
            gt_full = np.array(Image.open(os.path.join(GT, name + '.png')))
            img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt_full, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

            t = img.astype(np.float32) / 255.
            t = (t - MEAN) / STD
            x = torch.from_numpy(t.transpose(2, 0, 1)).unsqueeze(0).cuda()
            with torch.no_grad():
                _, _, conv5 = model(x, return_feat=True)
            f = conv5[0].cpu().numpy().astype(np.float32)   # (1024, Hf, Wf)
            Hf, Wf = f.shape[1], f.shape[2]

            bfrac, vf, maj = cell_stats(gt, Hf, Wf)
            rg = rgb_grad_mag(img).reshape(Hf, SIZE // Hf, Wf, SIZE // Wf).mean(axis=(1, 3))

            fg = feat_grad_mag(f)
            r_f, cb, ci = boundary_ratio(fg, bfrac, vf)
            b_ratio_f.append(r_f)
            n_bcell += cb; n_icell += ci
            r_rgb, _, _ = boundary_ratio(rg, bfrac, vf)
            b_ratio_rgb.append(r_rgb)

            # sample cells for probe 2 (valid, majority label)
            mask = (vf >= 0.5)
            sel = np.argwhere(mask)
            if len(sel) > 300:
                idx = np.random.RandomState(0).choice(len(sel), 300, replace=False)
                sel = sel[idx]
            for i, j in sel:
                F_list.append(f[:, i, j])
                y_list.append(maj[i, j])
        del model
        torch.cuda.empty_cache()

        F = np.stack(F_list).astype(np.float32)
        y = np.array(y_list)
        n_train = int(0.8 * len(y))
        rng = np.random.RandomState(0)
        perm = rng.permutation(len(y))
        F, y = F[perm], y[perm]
        Xtr, ytr, Xte, yte = F[:n_train], y[:n_train], F[n_train:], y[n_train:]

        # linear probe
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(Xtr, ytr)
        lin_acc = 100 * clf.score(Xte, yte)

        # nearest class prototype (cosine)
        cent = np.stack([Xtr[ytr == c].mean(axis=0) for c in range(21)])
        cent = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-8)
        Xte_n = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
        sim = Xte_n @ cent.T
        proto_acc = 100 * (sim.argmax(axis=1) == yte).mean()

        br_f = np.nanmean(b_ratio_f)
        br_rgb = np.nanmean(b_ratio_rgb)
        out(f'--- {tag} ({path}) ---')
        out(f'  cells: boundary={n_bcell}  interior={n_icell}')
        out(f'  boundary/interior gradient ratio:  conv5 = {br_f:.2f}   RGB reference = {br_rgb:.2f}')
        out(f'  pixel classification: linear probe = {lin_acc:.1f}%   nearest-class-prototype = {proto_acc:.1f}%'
            f'   (chance {100/21:.1f}%)')
        out('')

    out('  Interpretation:')
    out('  - conv5 ratio ~1 => features do not sharpen at semantic boundaries (smooth).')
    out('  - linear >> prototype => classification space, not a similarity metric.')
    os.makedirs('analysis', exist_ok=True)
    with open('analysis/probe_analysis.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines) + '\n')
    print('\nSaved: analysis/probe_analysis.txt')


if __name__ == '__main__':
    main()
