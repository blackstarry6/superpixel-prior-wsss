"""
Top-K high-confidence sampling contrast probe (PRE-REGISTERED; see
3-对比实验结果.md, section "2026-09-07 补充②" for the decision thresholds).

Question: does PCC-style high-confidence CAM sampling rescue class
separability at the feature level (which superpixel averaging destroys)?
This measures INPUT-SIGNAL AVAILABILITY in frozen spaces, not training
mechanisms.

Arms (per feature space, samples pooled across images, ONE classification):
  (i)   superpixel-mean prototypes (SLIC rs20 recipe on 448 image,
        feat-grid mean, GT majority label, purity>=0.9, fg-majority only)
  (ii)  per-class top-K high-confidence CAM pixels (K in {1,5,10}% of the
        class's claimed pixels per image, min 1 max 100); eval label = GT
        at the pixel (ignore 255 skipped)
  (iii) random pixels from the same claimed sets, same per-class counts
  (iv)  top-K among GT-pure claimed pixels (GT == claimed, 1-based) —
        upper bound

Classifier protocol (aligned with probe-2): per-arm pooled samples,
RandomState(0) 80/20 split, train-split class means, cosine
nearest-centroid, 21-way; all test sets are fg-labeled.

Stages:
  0 — smoke test (3 images, counts only)
  1 — extract baseline conv5 features (448 input) ->
      analysis/fm_feat/conv5_baseline/{name}.npy (float16)
  2 — full four-arm analysis -> analysis/topk_probe.txt

Usage: python analyze_topk_probe.py --stage {0,1,2} [--n 300]
"""
import argparse
import os
import sys

import numpy as np
import cv2
import torch
from PIL import Image

sys.path.insert(0, '.')
from spfr import generate_superpixels, resize_superpixels

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
GT_DIR = 'VOC2012/SegmentationClass'
JPEG = 'VOC2012/JPEGImages'
CAM_DIR = 'cam_baseline_new'
SIZE = 448
KS = [1, 5, 10]                # percent of per-image per-class claimed pixels
K_CAP = 100
K_MIN = 1
MIN_CELLS = 4                  # min feat-grid cells for an arm-(i) prototype
PURITY = 0.9

SPACES = [('conv5', 'analysis/fm_feat/conv5_baseline/'),
          ('clip', 'analysis/fm_feat/clip_vitb16/'),
          ('dino', 'analysis/fm_feat/dinov2_vitb14/')]

RNG0 = np.random.RandomState(0)   # split
RNG1 = np.random.RandomState(1)   # random-pixel arm


def load_names(n):
    names = [l.split(' ')[0].strip().split('/')[-1][:-4]
             for l in open('voc12/val.txt').read().strip().split('\n')]
    return names[:n]


def img448_tensor(name):
    img = np.array(Image.open(os.path.join(JPEG, name + '.jpg')).convert('RGB'))
    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    t = img.astype(np.float32) / 255.
    t = (t - MEAN) / STD
    return torch.from_numpy(t.transpose(2, 0, 1))


def cam_stack(name):
    d = np.load(os.path.join(CAM_DIR, name + '.npy'), allow_pickle=True).item()
    h, w = list(d.values())[0].shape
    fg = np.zeros((20, h, w), np.float32)
    for k, v in d.items():
        fg[int(k)] = v
    return fg


def classify(samples):
    """samples: list of (vec, label, img_idx). Pooled 80/20 RandomState(0)
    split, train class means, cosine nearest centroid, 21-way."""
    F = np.stack([np.asarray(s[0], np.float32) for s in samples])
    y = np.array([s[1] for s in samples])
    ok = np.linalg.norm(F, axis=1) > 1e-8
    F, y = F[ok], y[ok]
    perm = RNG0.permutation(len(y))
    n_tr = int(0.8 * len(y))
    tr, te = perm[:n_tr], perm[n_tr:]
    Ftr, ytr, Fte, yte = F[tr], y[tr], F[te], y[te]
    cent = np.stack([Ftr[ytr == c].mean(axis=0) if (ytr == c).any()
                     else np.zeros(F.shape[1], np.float32)
                     for c in range(21)])
    cent /= (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-8)
    Fte_n = Fte / (np.linalg.norm(Fte, axis=1, keepdims=True) + 1e-8)
    pred = (Fte_n @ cent.T).argmax(axis=1)
    acc = 100.0 * (pred == yte).mean()
    cnt = np.bincount(yte, minlength=21)
    cnt = cnt[cnt > 0]
    return acc, len(tr), len(te), cnt.min(), int(np.median(cnt))


def classify_byimage(samples):
    """Post-hoc robustness (NOT pre-registered): split by IMAGE (80/20
    images) so no feature cell can appear in both splits — kills the
    twin-sample leakage of the sample-level split."""
    F = np.stack([np.asarray(s[0], np.float32) for s in samples])
    y = np.array([s[1] for s in samples])
    im = np.array([s[2] for s in samples])
    ok = np.linalg.norm(F, axis=1) > 1e-8
    F, y, im = F[ok], y[ok], im[ok]
    imgs = np.unique(im)
    perm = RNG0.permutation(len(imgs))
    tr_im = set(imgs[perm[:int(0.8 * len(imgs))]].tolist())
    m_tr = np.isin(im, list(tr_im))
    Ftr, ytr, Fte, yte = F[m_tr], y[m_tr], F[~m_tr], y[~m_tr]
    cent = np.stack([Ftr[ytr == c].mean(axis=0) if (ytr == c).any()
                     else np.zeros(F.shape[1], np.float32)
                     for c in range(21)])
    cent /= (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-8)
    Fte_n = Fte / (np.linalg.norm(Fte, axis=1, keepdims=True) + 1e-8)
    pred = (Fte_n @ cent.T).argmax(axis=1)
    return 100.0 * (pred == yte).mean(), int(m_tr.sum()), int((~m_tr).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, required=True, choices=[0, 1, 2])
    ap.add_argument('--n', type=int, default=300)
    args = ap.parse_args()

    if args.stage == 1:
        from network.resnet38_SEAM import Net
        model = Net().cuda().eval()
        sd = torch.load('../SEAM_model/resnet38_SEAM.pth', map_location='cpu')
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd)
        names = load_names(args.n)
        os.makedirs('analysis/fm_feat/conv5_baseline', exist_ok=True)
        for i, name in enumerate(names):
            with torch.no_grad():
                _, _, conv5 = model(img448_tensor(name).unsqueeze(0).cuda(),
                                    return_feat=True)
            np.save(f'analysis/fm_feat/conv5_baseline/{name}.npy',
                    conv5[0].cpu().numpy().astype(np.float16))
            if (i + 1) % 50 == 0:
                print(f'  {i+1}/{len(names)}')
        print('stage 1 done:', len(names), 'files')
        return

    names = load_names(3 if args.stage == 0 else args.n)
    S = {}                       # (space, tag, K) -> [(vec float16, label)]
    prec5 = []                   # CAM-label precision of K=5% top pixels
    smoke_counts = []

    for ni, name in enumerate(names):
        gt = np.array(Image.open(os.path.join(GT_DIR, name + '.png')))
        fg = cam_stack(name)
        h, w = fg.shape[1], fg.shape[2]
        claim = fg.argmax(axis=0)                 # (h,w) 0-based fg class
        score = fg.max(axis=0)
        claimed = score > 0
        gtr = gt.ravel()
        seg448, _ = generate_superpixels(img448_tensor(name).cpu(),
                                         n_segments=300, region_size=20)

        for space, fdir in SPACES:
            fp = os.path.join(fdir, name + '.npy')
            if not os.path.exists(fp):
                continue
            feat = np.load(fp).astype(np.float32)      # (C,Hf,Wf)
            C, Hf, Wf = feat.shape
            yy, xx = np.mgrid[0:h, 0:w]
            cell = (np.minimum((yy * Hf) // h, Hf - 1) * Wf
                    + np.minimum((xx * Wf) // w, Wf - 1)).ravel()
            Fmap = feat.reshape(C, -1)

            seg = resize_superpixels(seg448, Hf, Wf).cpu().numpy().ravel()
            gtf = cv2.resize(gt, (Wf, Hf),
                             interpolation=cv2.INTER_NEAREST).ravel()

            # arm (i): superpixel-mean prototypes (fg-majority, purity>=0.9)
            protos = []
            for s in np.unique(seg):
                m = seg == s
                if m.sum() < MIN_CELLS:
                    continue
                lab = gtf[m]
                valid = lab[lab < 255]
                if len(valid) == 0:
                    continue
                bc = np.bincount(valid, minlength=21)
                c = int(bc.argmax())
                if c == 0 or bc.max() / len(valid) < PURITY:
                    continue
                protos.append((Fmap[:, m].mean(axis=1).astype(np.float16),
                               c, ni))
            S.setdefault((space, 'i', 0), []).extend(protos)

            # arms (ii)/(iii)/(iv); CAM keys 0-based, GT 1-based
            for K in KS:
                s_ii, s_iii, s_iv = [], [], []
                for c in range(20):
                    pix = np.argwhere((claimed & (claim == c)).ravel()).ravel()
                    if len(pix) == 0:
                        continue
                    nn = int(np.clip(round(K / 100.0 * len(pix)),
                                     K_MIN, K_CAP))
                    order = np.argsort(-score.ravel()[pix])
                    top = pix[order[:nn]]
                    gtc = gtr[top]
                    keep = gtc < 255
                    for p, g in zip(top[keep], gtc[keep]):
                        s_ii.append((Fmap[:, cell[p]].astype(np.float16),
                                     int(g), ni))
                    if K == 5:
                        prec5.extend(gtc == (c + 1))
                    rnd = pix[RNG1.choice(len(pix), min(nn, len(pix)),
                                          replace=False)]
                    gr = gtr[rnd]
                    kr = gr < 255
                    for p, g in zip(rnd[kr], gr[kr]):
                        s_iii.append((Fmap[:, cell[p]].astype(np.float16),
                                      int(g), ni))
                    pure = pix[gtr[pix] == (c + 1)]
                    if len(pure) > 0:
                        po = np.argsort(-score.ravel()[pure])
                        for p in pure[po[:nn]]:
                            s_iv.append((Fmap[:, cell[p]].astype(np.float16),
                                         c + 1, ni))
                if args.stage == 0:
                    smoke_counts.append((name, space, K, len(s_ii),
                                         len(s_iii), len(s_iv)))
                else:
                    S.setdefault((space, 'ii', K), []).extend(s_ii)
                    S.setdefault((space, 'iii', K), []).extend(s_iii)
                    S.setdefault((space, 'iv', K), []).extend(s_iv)

    if args.stage == 0:
        for r in smoke_counts:
            print(f'  {r[0]} {r[1]:5s} K={r[2]}%: ii={r[3]} iii={r[4]} '
                  f'iv={r[5]}')
        n_i = [k for k in S if k[1] == 'i']
        print(f'  arm-(i) prototypes pooled: '
              f'{ {k[0]: len(S[k]) for k in n_i} }')
        print('smoke OK')
        return

    out_lines = []

    def out(s=''):
        print(s)
        out_lines.append(s)

    out('=' * 70)
    out('  Top-K high-confidence sampling probe (pre-registered)')
    out('=' * 70)
    out(f'  images: {len(names)} (val first-n); pooled per (space,arm,K), '
        f'ONE classification each;')
    out(f'  classifier: cosine nearest-centroid (train-split class means), '
        f'21-way, 80/20 RandomState(0); chance ~4.8%')
    out(f'  arm (i): SLIC rs20 superpixel means, fg-majority, '
        f'purity>={PURITY}, >={MIN_CELLS} cells')
    out(f'  K sweep: {KS}% (cap {K_CAP}/class/image); '
        f'CAM precision of K=5% top pixels: {100*np.mean(prec5):.1f}%')
    out('')

    acc = {}
    for key in sorted(S, key=lambda k: (k[0],
                                        {'i': 0, 'ii': 1, 'iii': 2,
                                                         'iv': 3}[k[1]],
                                        k[2])):
        space, tag, K = key
        a, n_tr, n_te, cmin, cmed = classify(S[key])
        acc[key] = a
        out(f'  {space:5s} arm({tag:4s}) K={K if K else "-":>2}%: '
            f'acc={a:6.1f}%  n_train={n_tr} n_test={n_te} '
            f'per-class test min/med={cmin}/{cmed}')
    out('')

    out('  Pre-registered verdicts (K=5%, thresholds archived 2026-09-07):')
    for space, _ in SPACES:
        a_i = acc.get((space, 'i', 0), np.nan)
        a_ii = acc.get((space, 'ii', 5), np.nan)
        a_iii = acc.get((space, 'iii', 5), np.nan)
        if np.isnan(a_i) or np.isnan(a_ii):
            out(f'    {space}: insufficient samples')
            continue
        r = a_ii / max(a_i, 1e-8)
        if r >= 2 and a_ii >= 24:
            v = 'RESCUED'
        elif r < 1.5:
            v = 'NOT RESCUED'
        else:
            v = 'BOUNDARY'
        if (not np.isnan(a_iii)) and r >= 2 and a_ii / max(a_iii, 1e-8) < 1.2:
            v += ' [dont-average branch]'
        out(f'    {space}: (ii)/(i)={r:5.2f}  (ii)={a_ii:5.1f}%  '
            f'(iii)={a_iii:5.1f}%  -> {v}')
    out('')
    out('  Post-hoc robustness (NOT pre-registered): image-level 80/20 split')
    out('  (kills twin-sample leakage: adjacent top-K pixels share feature '
        'cells,)')
    out('  (so the sample-level split can place identical vectors in train '
        'and test.):')
    for key in sorted(S, key=lambda k: (k[0],
                                        {'i': 0, 'ii': 1, 'iii': 2,
                                                         'iv': 3}[k[1]],
                                        k[2])):
        space, tag, K = key
        a, n_tr, n_te = classify_byimage(S[key])
        out(f'  {space:5s} arm({tag:4s}) K={K if K else "-":>2}%: '
            f'acc={a:6.1f}%  (train {n_tr} / test {n_te} samples, '
            f'disjoint images)')
    out('')
    out('  Reading limits (pre-registered): frozen-space signal')
    out('  availability only; positive = "data-side hypothesis gains')
    out('  feature-level support", NOT "PCC mechanism confirmed";')
    out('  arm (iv) is an upper bound (center bias / part confusion).')
    os.makedirs('analysis', exist_ok=True)
    with open('analysis/topk_probe.txt', 'w', encoding='utf-8') as fo:
        fo.write('\n'.join(out_lines) + '\n')
    print('\nSaved: analysis/topk_probe.txt')


if __name__ == '__main__':
    main()
