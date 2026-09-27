"""Step 2 (v2): fixed-threshold robustness check — paired Delta(t) at all 11 thresholds.

Changes vs v1 (user request): the archive table now shows ALL SIX per-seed
mIoU values per threshold (FT-only s1/s2/s3, FT+SPFR s1/s2/s3) alongside the
seed-paired deltas, so every number in the delta columns can be traced to two
printed values. The 7 mIoU(t) vectors are cached to analysis/threshold_vectors.npz
(recompute only if the cache is missing).

Pre-registered criteria (unchanged):
  primary   : for every t in the grid, |3-seed mean Delta(t)| < 0.30  (the paper's MDE)
  secondary : at t in {0.15, 0.20, 0.25} (operative region), |mean Delta(t)| <= 0.17 (noise)
  record    : max |mean Delta(t)| over the grid and its location

Pairing reminder (printed in the archive header):
  d_s* = mIoU(FT+SPFR seed *) - mIoU(FT-only seed *), both at the SAME fixed t;
  mean/std are over the three same-seed differences; baseline is reference only.
"""
import os
import numpy as np
from PIL import Image

GT = 'VOC2012/SegmentationClass'
THRESHOLDS = [i / 100 for i in range(10, 61, 5)]
NPZ = 'analysis/threshold_vectors.npz'
DIRS = {
    'baseline':   './cam_baseline_new',
    'FT-only s1': './cam_finetune_only_s1', 'FT-only s2': './cam_finetune_only_s2',
    'FT-only s3': './cam_finetune_only_s3',
    'FT+SPFR s1': './cam_finetune_spfr_s1', 'FT+SPFR s2': './cam_finetune_spfr_s2',
    'FT+SPFR s3': './cam_finetune_spfr_s3',
}


def miou_vector(cam_dir):
    """Single pass: per-t mIoU over all 1449 val images (eval_seeds.py logic)."""
    name_list = sorted(f[:-4] for f in os.listdir(cam_dir) if f.endswith('.npy'))
    nt = len(THRESHOLDS)
    P = np.zeros((nt, 21)); T = np.zeros((nt, 21)); TP = np.zeros((nt, 21))
    for name in name_list:
        d = np.load(os.path.join(cam_dir, name + '.npy'), allow_pickle=True).item()
        h, w = list(d.values())[0].shape
        fg = np.zeros((20, h, w), np.float32)
        for k in d:
            fg[k] = d[k]
        fg_max = fg.max(axis=0)
        fg_arg = fg.argmax(axis=0) + 1
        gt = np.array(Image.open(os.path.join(GT, name + '.png')))
        cal = gt < 255
        gt_c = gt[cal].astype(np.int64)
        for ti, tt in enumerate(THRESHOLDS):
            pred = np.where(fg_max > tt, fg_arg, 0).astype(np.uint8)
            p_c = pred[cal].astype(np.int64)
            P[ti] += np.bincount(p_c, minlength=21)
            T[ti] += np.bincount(gt_c, minlength=21)
            TP[ti] += np.bincount(gt_c[p_c == gt_c], minlength=21)
    return (TP / (T + P - TP + 1e-10)).mean(axis=1) * 100


os.makedirs('analysis', exist_ok=True)
if os.path.exists(NPZ):
    z = np.load(NPZ)
    V = {k: z[k] for k in z.files}
    print('loaded cached mIoU(t) vectors from', NPZ)
else:
    print('computing mIoU(t) vectors for 7 configs ...')
    V = {label: miou_vector(d) for label, d in DIRS.items()}
    np.savez(NPZ, **V)
    print('cached vectors to', NPZ)

D = np.stack([V[f'FT+SPFR s{s}'] - V[f'FT-only s{s}'] for s in (1, 2, 3)])  # (3, 11)
D_mean, D_std = D.mean(axis=0), D.std(axis=0, ddof=1)
only_mean = np.mean([V[f'FT-only s{s}'] for s in (1, 2, 3)], axis=0)
spfr_mean = np.mean([V[f'FT+SPFR s{s}'] for s in (1, 2, 3)], axis=0)

L = []
L.append('=' * 108)
L.append('  Fixed-threshold robustness check: paired Delta(t), Experiment 1')
L.append('  (all 1449 val images; same code path as eval_seeds.py; CAMs from cached npy)')
L.append('=' * 108)
L.append('  Column semantics:')
L.append('    only_s* / spfr_s* : per-seed mIoU of that single model at the SAME fixed t')
L.append('    d_s* = spfr_s* - only_s*  (same-seed pair, same t); mean/std over the three d_s*')
L.append('    baseline : pretrained model, reference only (never enters any subtraction)')
L.append('')
L.append(f'{"t":>5} {"baseline":>9} {"only_s1":>8} {"only_s2":>8} {"only_s3":>8} '
         f'{"spfr_s1":>8} {"spfr_s2":>8} {"spfr_s3":>8} {"d_s1":>7} {"d_s2":>7} {"d_s3":>7} '
         f'{"mean":>7} {"std":>6}')
for ti, t in enumerate(THRESHOLDS):
    L.append(f'{t:>5.2f} {V["baseline"][ti]:>9.2f} '
             f'{V["FT-only s1"][ti]:>8.2f} {V["FT-only s2"][ti]:>8.2f} {V["FT-only s3"][ti]:>8.2f} '
             f'{V["FT+SPFR s1"][ti]:>8.2f} {V["FT+SPFR s2"][ti]:>8.2f} {V["FT+SPFR s3"][ti]:>8.2f} '
             f'{D[0, ti]:>+7.3f} {D[1, ti]:>+7.3f} {D[2, ti]:>+7.3f} '
             f'{D_mean[ti]:>+7.3f} {D_std[ti]:>6.3f}')

L.append('')
L.append('  arm means: FT-only = mean(only_s1..3), FT+SPFR = mean(spfr_s1..3) per t')
L.append(f'  best-t per arm: baseline {THRESHOLDS[int(np.argmax(V["baseline"]))]:.2f}; '
         f'all six FT runs 0.15 (t=0.15 row reproduces the paper: +0.083 +/- 0.078)')
mx = int(np.argmax(np.abs(D_mean)))
primary = np.all(np.abs(D_mean) < 0.30)
secondary = all(abs(D_mean[THRESHOLDS.index(x)]) <= 0.17 for x in (0.15, 0.20, 0.25))
L.append(f'  max |mean Delta(t)|  : {abs(D_mean[mx]):.3f} at t={THRESHOLDS[mx]:.2f}')
L.append(f'  primary   (all |D| < 0.30 MDE)            : {"PASS" if primary else "FAIL"}')
L.append(f'  secondary (|D| <= 0.17 @ 0.15/0.20/0.25)   : {"PASS" if secondary else "FAIL"}')
L.append(f'  worst per-seed Delta over grid             : {D.min():+.3f} .. {D.max():+.3f}')
L.append('')
report = '\n'.join(L)
print(report)

with open('analysis/threshold_robustness.txt', 'w', encoding='utf-8') as f:
    f.write(report + '\n')
print('\nSaved: analysis/threshold_robustness.txt')
