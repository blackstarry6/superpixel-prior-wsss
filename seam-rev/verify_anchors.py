"""Anchor verification for the fixed-threshold robustness check (Step 1).

Re-runs the exact evaluate_dir logic of eval_seeds.py (single pass, 11-threshold
confusion accumulation, strict '>' rule) on the 7 caches and verifies the
archived anchor values BEFORE any new statistics are trusted:

  baseline @ best t        = 51.23 (t=0.25)
  FT-only  s1/s2/s3        = 51.16 / 51.06 / 51.40 (t=0.15)
  FT+SPFR  s1/s2/s3        = 51.23 / 51.07 / 51.56 (t=0.15)
  per-seed paired delta    = +0.07 / +0.01 / +0.17   (full precision, +-0.011 tol)
  mean delta               = +0.08 +- 0.08
  person paired per-seed   = +0.63 / +0.68 / +0.78, mean +0.69 (secondary)
"""
import os
import numpy as np
from PIL import Image

GT = 'VOC2012/SegmentationClass'
THRESHOLDS = [i / 100 for i in range(10, 61, 5)]
CATEGORIES = ['bg', 'aero', 'bike', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
              'chair', 'cow', 'table', 'dog', 'horse', 'motor', 'person', 'plant',
              'sheep', 'sofa', 'train', 'tv']
PERSON = CATEGORIES.index('person')

DIRS = {
    'baseline':     './cam_baseline_new',
    'FT-only s1':   './cam_finetune_only_s1',
    'FT-only s2':   './cam_finetune_only_s2',
    'FT-only s3':   './cam_finetune_only_s3',
    'FT+SPFR s1':   './cam_finetune_spfr_s1',
    'FT+SPFR s2':   './cam_finetune_spfr_s2',
    'FT+SPFR s3':   './cam_finetune_spfr_s3',
}

ANCHOR_MIOU = {'baseline': (51.23, 0.25), 'FT-only s1': (51.16, 0.15),
               'FT-only s2': (51.06, 0.15), 'FT-only s3': (51.40, 0.15),
               'FT+SPFR s1': (51.23, 0.15), 'FT+SPFR s2': (51.07, 0.15),
               'FT+SPFR s3': (51.56, 0.15)}


def evaluate_dir(cam_dir):
    """Exact copy of eval_seeds.py logic; returns per-t mIoU vector + per-class at best."""
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
    iou = TP / (T + P - TP + 1e-10)
    miou_all = iou.mean(axis=1) * 100
    best_ti = int(np.argmax(miou_all))
    return miou_all, THRESHOLDS[best_ti], iou[best_ti] * 100


results = {}
print(f'{"config":<12}{"best mIoU":>10}{"best_t":>8}   anchor          verdict')
all_ok = True
for label, cam_dir in DIRS.items():
    miou_all, best_t, per_class = evaluate_dir(cam_dir)
    results[label] = (miou_all, best_t, per_class)
    a_m, a_t = ANCHOR_MIOU[label]
    ok = (round(miou_all.max(), 2) == a_m) and (abs(best_t - a_t) < 1e-9)
    all_ok &= ok
    print(f'{label:<12}{miou_all.max():>10.4f}{best_t:>8.2f}   {a_m:.2f}@{a_t:.2f}     {"PASS" if ok else "FAIL"}')

# paired deltas (full precision)
deltas, person_d = [], []
for s in (1, 2, 3):
    d = results[f'FT+SPFR s{s}'][0].max() - results[f'FT-only s{s}'][0].max()
    deltas.append(d)
    person_d.append(results[f'FT+SPFR s{s}'][2][PERSON] - results[f'FT-only s{s}'][2][PERSON])
print()
print('per-seed delta (full precision):', [f'{d:+.4f}' for d in deltas])
anchor_d = [0.07, 0.01, 0.17]
ok_d = all(abs(d - a) <= 0.011 for d, a in zip(deltas, anchor_d))
print(f'mean delta = {np.mean(deltas):+.4f} (anchor +0.08 +- 0.08)  ->', 'PASS' if abs(np.mean(deltas) - 0.08) < 0.005 else 'FAIL')
print('per-seed anchor match:', 'PASS' if ok_d else 'FAIL')
print('person per-seed delta:', [f'{d:+.4f}' for d in person_d],
      f'mean {np.mean(person_d):+.4f} (anchor +0.63/+0.68/+0.78, mean +0.69)')
ok_p = all(abs(d - a) <= 0.011 for d, a in zip(person_d, [0.63, 0.68, 0.78]))
print('person anchor match:', 'PASS' if ok_p else 'FAIL')

print()
print('=== STEP 1 VERDICT:', 'ALL PASS' if (all_ok and ok_d and ok_p) else 'CHECK FAILURES ABOVE', '===')
