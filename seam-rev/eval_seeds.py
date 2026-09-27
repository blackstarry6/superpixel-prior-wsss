"""
Evaluate CAM mIoU for the 6 seed runs of Experiment 1 (3-seed paired).

Single-pass threshold scan: per image, compute fg_max/fg_argmax ONCE, then
derive predictions for all 11 thresholds by comparison
(pred = fg_class if fg_max > t else 0, equivalent to argmax over the
21-channel tensor with t[0]=t; ties go to background, matching the original).

Usage: python eval_seeds.py
Output: scan_results_seeds.txt (new file; scan_results.txt untouched)
"""
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

GT = 'VOC2012/SegmentationClass'

cam_dirs = {
    'FT-only  s1': './cam_finetune_only_s1',
    'FT+SPFR  s1': './cam_finetune_spfr_s1',
    'FT-only  s2': './cam_finetune_only_s2',
    'FT+SPFR  s2': './cam_finetune_spfr_s2',
    'FT-only  s3': './cam_finetune_only_s3',
    'FT+SPFR  s3': './cam_finetune_spfr_s3',
    # ── H2 intervention arm (added 2026-09-05) ──
    # SPFR on 128-d projection head; only evaluated if the directory exists.
    'FT+SPFRproj s1': './cam_finetune_spfr_proj_s1',
    'FT+SPFRproj s2': './cam_finetune_spfr_proj_s2',
    'FT+SPFRproj s3': './cam_finetune_spfr_proj_s3',
}
BASELINE_MIOU = 51.23

categories = ['bg', 'aero', 'bike', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'table', 'dog', 'horse', 'motor', 'person', 'plant', 'sheep', 'sofa', 'train', 'tv']
PERSON_IDX = categories.index('person')

THRESHOLDS = [i / 100 for i in range(10, 61, 5)]  # 0.10 ~ 0.60 step 0.05


def evaluate_dir(cam_dir):
    """Return (miou, best_t, per_class) for one cam directory (single pass)."""
    name_list = sorted(f[:-4] for f in os.listdir(cam_dir) if f.endswith('.npy'))
    if not name_list:
        return None

    nt = len(THRESHOLDS)
    P = np.zeros((nt, 21))
    T = np.zeros((nt, 21))
    TP = np.zeros((nt, 21))

    for name in tqdm(name_list, desc=os.path.basename(cam_dir)):
        d = np.load(os.path.join(cam_dir, name + '.npy'), allow_pickle=True).item()
        h, w = list(d.values())[0].shape
        fg = np.zeros((20, h, w), np.float32)
        for k in d:
            fg[k] = d[k]                      # 0-based key k -> class k+1
        fg_max = fg.max(axis=0)
        fg_arg = fg.argmax(axis=0) + 1        # class index 1..20

        gt = np.array(Image.open(os.path.join(GT, name + '.png')))
        cal = gt < 255
        gt_c = gt[cal].astype(np.int64)
        for ti, tt in enumerate(THRESHOLDS):
            pred = np.where(fg_max > tt, fg_arg, 0).astype(np.uint8)
            p_c = pred[cal].astype(np.int64)
            P[ti] += np.bincount(p_c, minlength=21)
            T[ti] += np.bincount(gt_c, minlength=21)
            TP[ti] += np.bincount(gt_c[p_c == gt_c], minlength=21)

    miou_all = np.mean(TP / (T + P - TP + 1e-10), axis=1) * 100
    best_ti = int(np.argmax(miou_all))
    per_class = TP[best_ti] / (T[best_ti] + P[best_ti] - TP[best_ti] + 1e-10) * 100
    return miou_all[best_ti], THRESHOLDS[best_ti], per_class


def main():
    results = {}
    for label, cam_dir in cam_dirs.items():
        if not os.path.isdir(cam_dir):
            print(f'{label}: dir not found, skip')
            continue
        r = evaluate_dir(cam_dir)
        if r is None:
            print(f'{label}: no npy files, skip')
            continue
        results[label] = {'miou': r[0], 'best_t': r[1], 'per_class': r[2]}
        print(f'{label}: mIoU={r[0]:.2f}% (best_t={r[1]:.2f}, {len(os.listdir(cam_dir))} npy)')

    if not results:
        print('no results — run phase 2 (infer_epoch.py) first')
        return

    lines = []
    def out(s=''):
        print(s)
        lines.append(s)

    out('=' * 65)
    out('  EXPERIMENT 1 — CAM mIoU (3 seeds, best threshold, no CRF)')
    out('=' * 65)
    out(f'{"Model":12s} {"mIoU":>7s} {"best_t":>7s} {"vs FT-only (paired)":>20s}')
    for seed in (1, 2, 3):
        only = results.get(f'FT-only  s{seed}')
        spfr = results.get(f'FT+SPFR  s{seed}')
        if only and spfr:
            delta = spfr['miou'] - only['miou']
            out(f'{"FT-only " + str(seed):12s} {only["miou"]:6.2f}% {only["best_t"]:6.2f}  {"":20s}')
            out(f'{"FT+SPFR " + str(seed):12s} {spfr["miou"]:6.2f}% {spfr["best_t"]:6.2f}  {delta:+.2f}%')
    out(f'{"Baseline":12s} {BASELINE_MIOU:6.2f}% (reference, infer_SEAM.py protocol)')

    # Paired statistics across seeds
    miou_deltas = []
    person_deltas = []
    for seed in (1, 2, 3):
        only = results.get(f'FT-only  s{seed}')
        spfr = results.get(f'FT+SPFR  s{seed}')
        if only and spfr:
            miou_deltas.append(spfr['miou'] - only['miou'])
            person_deltas.append(spfr['per_class'][PERSON_IDX] - only['per_class'][PERSON_IDX])
    out('')
    if miou_deltas:
        d = np.array(miou_deltas)
        out(f'Delta mIoU (SPFR - FT-only): {d.mean():+.2f}% +/- {d.std(ddof=1):.2f}%  (per-seed: {", ".join(f"{x:+.2f}" for x in d)})')
    if person_deltas:
        p = np.array(person_deltas)
        out(f'Person net effect        : {p.mean():+.2f}% +/- {p.std(ddof=1):.2f}%  (per-seed: {", ".join(f"{x:+.2f}" for x in p)})')

    # Per-class table at best threshold
    out('')
    out('=' * 65)
    out('  Per-Class mIoU (best threshold)')
    out('=' * 65)
    header = f'{"Class":10s}'
    for label in results:
        header += f' {label[:10]:>10s}'
    out(header)
    out('-' * (10 + 11 * len(results)))
    for i in range(21):
        row = f'{categories[i]:10s}'
        for label in results:
            row += f' {results[label]["per_class"][i]:9.2f}%'
        out(row)

    with open('scan_results_seeds.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nSaved: scan_results_seeds.txt')


if __name__ == '__main__':
    main()
