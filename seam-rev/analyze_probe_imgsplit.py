"""probe-2 leakage audit: image-level split re-evaluation (NEW file; the
originals analyze_probe.py / analyze_probe_fm.py are NOT modified and their
outputs are NOT overwritten).

Motivation: the top-K probe (2026-09-07) showed that sample-level 80/20
splits can inflate accuracies when train and test share images (FM spaces:
12.7%/21.2% collapsed to 2.3%/3.2% under an image-level split). probe-2 uses
the same sample-level split family, so the published numbers
(conv5 85.3/66.2, CLIP 71.3/10.2, DINOv2 70.6/4.2) are re-audited here.

Protocol replication (verbatim from analyze_probe_fm.py stage 2 /
analyze_probe.py):
  - cells via cell_stats (imported from analyze_probe), vf>=0.5,
    cap 300 cells/image via np.random.RandomState(0).choice;
  - ANCHOR split (sample-level) must reproduce the published numbers;
  - AUDIT split (image-level): RandomState(0) permutation of the 300 images,
    240/60, all cells of a test image held out;
  - classifiers identical: RidgeClassifier(alpha=1.0); nearest-class-
    prototype = train class means + cosine.

Features: conv5 from analysis/fm_feat/conv5_baseline (float16 cache from
the top-K probe; original probe used fresh float32 — small rounding
differences possible, reported as anchor deltas); CLIP/DINOv2 from their
caches.

Output: analysis/probe2_imgsplit.txt
Usage:  python analyze_probe_imgsplit.py
"""
import os
import sys

import numpy as np
import cv2
from PIL import Image
from sklearn.linear_model import RidgeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_probe import cell_stats

SIZE = 448
N_IMAGES = 300
GT = 'VOC2012/SegmentationClass'
SPACES = [('conv5', 'analysis/fm_feat/conv5_baseline'),
          ('clip_vitb16', 'analysis/fm_feat/clip_vitb16'),
          ('dinov2_vitb14', 'analysis/fm_feat/dinov2_vitb14')]
PUBLISHED = {   # (linear, proto) from probe_analysis.txt / probe_fm.txt
    'conv5': (85.3, 66.2),
    'clip_vitb16': (71.3, 10.2),
    'dinov2_vitb14': (70.6, 4.2),
}


def collect_cells(fdir):
    """Exact replication of the probe-2 cell construction; returns
    (F, y, img_idx) pooled in val order."""
    names = [l.split()[0].strip().split('/')[-1][:-4]
             for l in open('voc12/val.txt').read().strip().split('\n')
             ][:N_IMAGES]
    F_list, y_list, im_list = [], [], []
    for ni, name in enumerate(names):
        gt = np.array(Image.open(os.path.join(GT, name + '.png')))
        gt = cv2.resize(gt, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        f = np.load(os.path.join(fdir, name + '.npy')).astype(np.float32)
        Hf, Wf = f.shape[1], f.shape[2]
        _, vf, maj = cell_stats(gt, Hf, Wf)
        mask = (vf >= 0.5)
        sel = np.argwhere(mask)
        if len(sel) > 300:
            idx = np.random.RandomState(0).choice(len(sel), 300,
                                                  replace=False)
            sel = sel[idx]
        for i, j in sel:
            F_list.append(f[:, i, j])
            y_list.append(maj[i, j])
            im_list.append(ni)
    return (np.stack(F_list).astype(np.float32), np.array(y_list),
            np.array(im_list))


def two_classifiers(Xtr, ytr, Xte, yte):
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(Xtr, ytr)
    lin = 100 * clf.score(Xte, yte)
    cent = np.stack([Xtr[ytr == c].mean(axis=0) for c in range(21)])
    cent = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-8)
    Xte_n = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
    proto = 100 * (np.argmax(Xte_n @ cent.T, axis=1) == yte).mean()
    return lin, proto


def main():
    out_lines = []

    def out(s=''):
        print(s)
        out_lines.append(s)

    out('=' * 72)
    out('  probe-2 leakage audit: image-level split re-evaluation')
    out('  (originals untouched; anchor must reproduce published numbers)')
    out('=' * 72)
    out('  cells: cell_stats vf>=0.5, cap 300/img (RandomState(0)); '
        'classifiers: ridge a=1 + cosine prototype (21-way)')
    out('  audit split: 300 images permuted (RandomState(0)), 240/60;')
    out('  all cells of a test image held out (same-image leakage '
        'impossible)')
    out('')
    out(f'  {"space":12s} {"split":7s} {"linear":>7s} {"proto":>7s} '
        f'{"n_tr":>6s} {"n_te":>6s}   note')
    for key, fdir in SPACES:
        F, y, im = collect_cells(fdir)

        # anchor: exact sample-level split from the originals
        n_train = int(0.8 * len(y))
        rng = np.random.RandomState(0)
        perm = rng.permutation(len(y))
        Fs, ys = F[perm], y[perm]
        lin_a, pr_a = two_classifiers(Fs[:n_train], ys[:n_train],
                                      Fs[n_train:], ys[n_train:])
        pl, pp = PUBLISHED[key]
        out(f'  {key:12s} {"sample":7s} {lin_a:7.1f} {pr_a:7.1f} '
            f'{n_train:6d} {len(y)-n_train:6d}   anchor '
            f'(published {pl}/{pp}; delta {lin_a-pl:+.1f}/{pr_a-pp:+.1f})')

        # audit: image-level split
        imgs = np.unique(im)
        rng2 = np.random.RandomState(0)
        perm2 = rng2.permutation(len(imgs))
        tr_im = set(imgs[perm2[:int(0.8 * len(imgs))]].tolist())
        m_tr = np.isin(im, list(tr_im))
        lin_b, pr_b = two_classifiers(F[m_tr], y[m_tr], F[~m_tr], y[~m_tr])
        out(f'  {key:12s} {"byimage":7s} {lin_b:7.1f} {pr_b:7.1f} '
            f'{int(m_tr.sum()):6d} {int((~m_tr).sum()):6d}   audit '
            f'(delta vs anchor {lin_b-lin_a:+.1f}/{pr_b-pr_a:+.1f})')
        out('')
    out('  Reading: if audit ~ anchor (<2 pts) the published numbers stand')
    out('  with a protocol note; if audit drops materially, the published')
    out('  numbers should be updated — the direction strengthens the')
    out('  premise-defect claim (prototype moves toward chance).')
    with open('analysis/probe2_imgsplit.txt', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out_lines) + '\n')
    print('\nSaved: analysis/probe2_imgsplit.txt')


if __name__ == '__main__':
    main()
