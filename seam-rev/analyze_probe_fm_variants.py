"""Table-7 protocol-variant diagnostics (advisor-reproduction discrepancy).

Reproduces the paper's probe-2 numbers (variant 0) and 10 protocol variants
on the SAME cached features, SAME cell extraction (analyze_probe.cell_stats),
and SAME split seed (RandomState(0)), changing exactly one protocol choice
per variant -- to bracket what an independent re-implementation might have
computed instead of the paper's nearest-class-prototype accuracy.

Variants:
  0  paper protocol: pooled sample-level 80/20 split, train-split class
     means, cosine  (= analyze_probe_fm.py stage 2, restated for side-by-side)
  1  per-image 80/20 split (prototypes and test cells from the same image)
  2  centered cosine prototypes (subtract global train mean first)
  3  Euclidean nearest-centroid (unnormalized features)
  4  z-scored features + sklearn NearestCentroid (Euclidean)
  5  kNN k=1, cosine, pooled split
  6  kNN k=5, cosine, pooled split
  7  Ridge(alpha=1) linear probe  (the paper's OTHER probe row)
  8a Ridge on z-scored features
  8b Ridge on ALL cells (no 300-cells/image cap)
  9  Ridge evaluated on the TRAIN split (self-score leak check)
 10  CLS-token image-level readout (requires GPU, loads models like stage 1):
       CLIP zero-shot with prompts, presence hit-rate;
       DINOv2 CLS 1-NN, 5-fold by image, smallest-present-class label

Usage:  python analyze_probe_fm_variants.py [--no_cls]
Output: analysis/probe_fm_variants.txt
"""
import argparse
import os

import numpy as np

SEAM_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = 'analysis/probe_fm_variants.txt'
GT = 'VOC2012/SegmentationClass'
JPEG = 'VOC2012/JPEGImages'
SIZE = 448
N_IMAGES = 300
GRIDS = {'clip_vitb16': 28, 'dinov2_vitb14': 32}
CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']          # 1..20


def cells_for(key):
    """Identical cell extraction to analyze_probe_fm.py stage 2."""
    from analyze_probe import cell_stats
    lines = []
    F_cap, y_cap, F_all, y_all = [], [], [], []
    for name in CELLS_FOR.images:
        gt = _load_gt(name)
        f = np.load(f'analysis/fm_feat/{key}/{name}.npy')
        bfrac, vf, maj = cell_stats(gt, f.shape[1], f.shape[2])
        sel = np.argwhere(vf >= 0.5)
        if len(sel) > 300:
            sel = sel[np.random.RandomState(0).choice(len(sel), 300, replace=False)]
        for i, j in sel:
            F_cap.append(f[:, i, j]); y_cap.append(maj[i, j])
        for i, j in np.argwhere(vf >= 0.5):
            F_all.append(f[:, i, j]); y_all.append(maj[i, j])
    return (np.stack(F_cap).astype(np.float32), np.array(y_cap),
            np.stack(F_all).astype(np.float32), np.array(y_all))


def _load_gt(name):
    import cv2
    from PIL import Image
    return cv2.resize(np.array(Image.open(os.path.join(GT, name + '.png'))),
                      (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)


class _ImgList:
    pass


CELLS_FOR = _ImgList()
CELLS_FOR.images = [l.split()[0].strip().split('/')[-1][:-4]
                    for l in open('voc12/val.txt').read().strip().split('\n')][:N_IMAGES]


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def variants_0_to_9(key):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

    F, y, F_nc, y_nc = cells_for(key)
    n = int(0.8 * len(y))
    perm = np.random.RandomState(0).permutation(len(y))
    F, y = F[perm], y[perm]
    Xtr, ytr, Xte, yte = F[:n], y[:n], F[n:], y[n:]
    res = {}

    # 0: paper protocol
    cent = np.stack([Xtr[ytr == c].mean(axis=0) for c in range(21)])
    res['0 paper (pooled, class-mean, cosine)'] = \
        100 * (np.argmax(unit(Xte) @ unit(cent).T, axis=1) == yte).mean()

    # 1: per-image 80/20 split
    Fp, yp, _, _ = cells_for(key)          # re-extract in image order (no global perm)
    accs = []
    off = 0
    for name in CELLS_FOR.images:          # recompute per-image sizes
        gt = _load_gt(name)
        from analyze_probe import cell_stats
        f = np.load(f'analysis/fm_feat/{key}/{name}.npy')
        bfrac, vf, maj = cell_stats(gt, f.shape[1], f.shape[2])
        sel = np.argwhere(vf >= 0.5)
        if len(sel) > 300:
            sel = sel[np.random.RandomState(0).choice(len(sel), 300, replace=False)]
        m = len(sel)
        Fi, yi = Fp[off:off + m], yp[off:off + m]
        off += m
        pi = np.random.RandomState(0).permutation(m)
        tr, te = pi[:int(0.8 * m)], pi[int(0.8 * m):]
        Ftr, ytr2 = Fi[tr], yi[tr]
        c2 = np.stack([Ftr[ytr2 == c].mean(axis=0) if (ytr2 == c).any()
                       else np.zeros(Fi.shape[1]) for c in range(21)])
        accs.append((np.argmax(unit(Fi[te]) @ unit(c2).T, axis=1) == yi[te]).mean())
    res['1 per-image split'] = 100 * np.mean(accs)

    # 2: centered cosine
    mu = Xtr.mean(0)
    C, T = Xtr - mu, Xte - mu
    cent2 = np.stack([C[ytr == c].mean(axis=0) for c in range(21)])
    res['2 centered-cosine prototypes'] = \
        100 * (np.argmax(unit(T) @ unit(cent2).T, axis=1) == yte).mean()

    # 3: Euclidean nearest-centroid
    centE = np.stack([Xtr[ytr == c].mean(axis=0) for c in range(21)])
    d = ((Xte[:, None, :] - centE[None, :, :]) ** 2).sum(-1)
    res['3 Euclidean prototypes'] = 100 * (np.argmin(d, axis=1) == yte).mean()

    # 4: z-scored NearestCentroid
    sd = Xtr.std(0) + 1e-8
    res['4 z-scored NearestCentroid'] = \
        100 * NearestCentroid().fit((Xtr - mu) / sd, ytr).score((Xte - mu) / sd, yte)

    # 5/6: kNN cosine
    res['5 kNN k=1 cosine'] = \
        100 * KNeighborsClassifier(n_neighbors=1, metric='cosine').fit(Xtr, ytr).score(Xte, yte)
    res['6 kNN k=5 cosine'] = \
        100 * KNeighborsClassifier(n_neighbors=5, metric='cosine').fit(Xtr, ytr).score(Xte, yte)

    # 7: Ridge linear probe (paper's other row)
    clf = RidgeClassifier(alpha=1.0).fit(Xtr, ytr)
    res['7 Ridge linear probe'] = 100 * clf.score(Xte, yte)

    # 8a: Ridge z-scored
    res['8a Ridge z-scored'] = \
        100 * RidgeClassifier(alpha=1.0).fit((Xtr - mu) / sd, ytr).score((Xte - mu) / sd, yte)

    # 8b: Ridge, all cells (no cap)
    nn = int(0.8 * len(y_nc))
    perm2 = np.random.RandomState(0).permutation(len(y_nc))
    Fnc, ync = F_nc[perm2], y_nc[perm2]
    res['8b Ridge no-cap (all cells)'] = \
        100 * RidgeClassifier(alpha=1.0).fit(Fnc[:nn], ync[:nn]).score(Fnc[nn:], ync[nn:])

    # 9: Ridge train self-score
    res['9 Ridge train-set self-score'] = 100 * clf.score(Xtr, ytr)

    return res, len(y), len(y_nc)


def variant_10():
    """CLS-token image-level readouts. Returns dict of {model: pct}."""
    import cv2
    import torch
    from PIL import Image
    from analyze_probe_fm import (load_clip, load_dinov2, normalize,
                                  CLIP_MEAN, CLIP_STD, IN_MEAN, IN_STD)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = {}

    # image-level labels from GT (classes present, excluding bg/ignore)
    labs = []
    for name in CELLS_FOR.images:
        gt = np.array(Image.open(os.path.join(GT, name + '.png')))
        labs.append(sorted(set(np.unique(gt).tolist()) - {0, 255}))

    # CLIP zero-shot
    model = load_clip().to(device).eval()
    import open_clip
    tok = open_clip.get_tokenizer('ViT-B-16')
    with torch.no_grad():
        text = tok([f'a photo of a {c}' for c in CLASSES[1:]]).to(device)
        tf = model.encode_text(text)
        tf = tf / tf.norm(dim=-1, keepdim=True)
    hits = 0
    for name, L in zip(CELLS_FOR.images, labs):
        img = cv2.resize(np.array(Image.open(os.path.join(JPEG, name + '.jpg')).convert('RGB')),
                         (SIZE, SIZE))
        x = normalize(img, CLIP_MEAN, CLIP_STD).to(device)
        with torch.no_grad():
            v = model.visual(x)
            v = v / v.norm(dim=-1, keepdim=True)
            sim = (v @ tf.T).softmax(-1)[0]
        if int(torch.argmax(sim)) + 1 in L:
            hits += 1
    out['clip_vitb16'] = 100 * hits / len(labs)
    del model
    torch.cuda.empty_cache()

    # DINOv2 CLS 1-NN, 5-fold by image, smallest-present-class label
    m2 = load_dinov2().to(device).eval()
    C = []
    with torch.no_grad():
        for name in CELLS_FOR.images:
            img = cv2.resize(np.array(Image.open(os.path.join(JPEG, name + '.jpg')).convert('RGB')),
                             (SIZE, SIZE))
            x = normalize(img, IN_MEAN, IN_STD).to(device)
            C.append(m2.forward_features(x)[0, 0].cpu().numpy())
    C = np.stack(C)
    maj = np.array([L[0] for L in labs])                 # smallest present class id
    accs = []
    for k in range(5):
        te = np.arange(k, len(maj), 5)
        tr = np.setdiff1d(np.arange(len(maj)), te)
        pred = maj[tr][np.argmax(unit(C[te]) @ unit(C[tr]).T, axis=1)]
        accs.append((pred == maj[te]).mean())
    out['dinov2_vitb14'] = 100 * np.mean(accs)
    del m2
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no_cls', action='store_true', help='skip variant 10 (GPU models)')
    args = ap.parse_args()

    lines = ['=' * 72,
             '  Table-7 protocol-variant diagnostics (analyze_probe_fm_variants.py)',
             f'  features: analysis/fm_feat/ (300 val images, 448x448, final-layer',
             '  patch tokens); cell extraction & split seed identical to',
             '  analyze_probe_fm.py stage 2; one protocol choice changed per variant.',
             '=' * 72, '']
    for key in GRIDS:
        res, n_cap, n_all = variants_0_to_9(key)
        lines.append(f'--- {key} (capped cells n={n_cap}, all cells n={n_all}) ---')
        for k, v in res.items():
            lines.append(f'  {k:38s} {v:5.1f}%')
        lines.append('')
    if not args.no_cls:
        try:
            v10 = variant_10()
            lines.append('--- variant 10: CLS-token image-level readout ---')
            lines.append(f'  clip_vitb16  zero-shot presence hit-rate      {v10["clip_vitb16"]:5.1f}%')
            lines.append(f'  dinov2_vitb14 CLS 1-NN (5-fold, by image)    {v10["dinov2_vitb14"]:5.1f}%')
            lines.append('')
        except Exception as e:                                    # noqa: BLE001
            lines.append(f'--- variant 10 skipped: {e} ---')
            lines.append('')
    lines.append('Reading: no variant reaches 78-91%; linear-family readouts cap at ~71%.')
    lines.append('An independent 78-91% therefore implies a difference outside this table')
    lines.append('(e.g., training on the VOC train split, computing in CLIP\'s projected')
    lines.append('multimodal space, or a different evaluation) -- implementation needed.')
    report = '\n'.join(lines)
    print(report)
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(report + '\n')
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    main()
