"""
Case-level failure analysis for Experiment 1 (classification stage).

For each val image: per-image mIoU for FT-only s{1,2,3} and FT+SPFR s{1,2,3}
at the archived optimal threshold t=0.15 (baseline at t=0.25), then
Damage(image) = mean over 3 seed pairs of (mIoU(SPFR) - mIoU(FT-only)).

Splits images by image-level co-occurrence (voc12/cls_labels.npy):
  single-object vs multi-object; person-alone vs person-co-occurring;
  specific pairs (person+bicycle, person+dog, cat+dog, ...).
Permutation test (numpy-only) for group differences.

Sanity check: global mIoU per config must reproduce scan_results_seeds.txt.

Outputs: analysis/case_damage_analysis.txt, analysis/per_image_stats.npz

Usage: python analyze_cases.py
"""
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

GT = 'VOC2012/SegmentationClass'
CLS_LABELS = 'voc12/cls_labels.npy'

cam_pairs = [
    ('cam_finetune_only_s1',  'cam_finetune_spfr_s1'),
    ('cam_finetune_only_s2',  'cam_finetune_spfr_s2'),
    ('cam_finetune_only_s3',  'cam_finetune_spfr_s3'),
]
BASELINE_DIR = 'cam_baseline_new'
T_RUN = 0.15      # optimal threshold for all 6 controlled runs (archived)
T_BASE = 0.25     # optimal threshold for baseline (archived)

categories = ['bg', 'aero', 'bike', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'table', 'dog', 'horse', 'motor', 'person', 'plant', 'sheep', 'sofa', 'train', 'tv']

# Reference values from scan_results_seeds.txt / archive §1.2 (sanity check)
EXPECTED_MIOU = {
    'cam_finetune_only_s1': 51.16, 'cam_finetune_spfr_s1': 51.23,
    'cam_finetune_only_s2': 51.06, 'cam_finetune_spfr_s2': 51.07,
    'cam_finetune_only_s3': 51.40, 'cam_finetune_spfr_s3': 51.56,
}

PAIRS = [
    ('person', 'bicycle'), ('person', 'dog'), ('person', 'cat'), ('person', 'horse'),
    ('person', 'motor'), ('person', 'bird'), ('person', 'boat'), ('person', 'bottle'),
    ('person', 'bus'), ('person', 'car'), ('person', 'chair'), ('person', 'cow'),
    ('person', 'table'), ('person', 'plant'), ('person', 'sheep'), ('person', 'sofa'),
    ('person', 'train'), ('person', 'tv'),
    ('cat', 'dog'), ('dog', 'horse'), ('bird', 'cat'),
]


def per_image_miou(cam, gt, t):
    """Per-image mIoU over classes with (T+P)>0, at threshold t."""
    h, w = list(cam.values())[0].shape
    fg = np.zeros((20, h, w), np.float32)
    for k in cam:
        fg[k] = cam[k]
    fg_max = fg.max(axis=0)
    fg_arg = fg.argmax(axis=0) + 1
    pred = np.where(fg_max > t, fg_arg, 0).astype(np.uint8)

    cal = gt < 255
    gt_c = gt[cal].astype(np.int64)
    p_c = pred[cal].astype(np.int64)
    T = np.bincount(gt_c, minlength=21)
    P = np.bincount(p_c, minlength=21)
    TP = np.bincount(gt_c[p_c == gt_c], minlength=21)
    valid = (T + P) > 0
    iou = TP[valid] / (T[valid] + P[valid] - TP[valid] + 1e-10)
    return iou.mean() * 100


def main():
    names = sorted(f[:-4] for f in os.listdir(cam_pairs[0][0]) if f.endswith('.npy'))
    print(f'{len(names)} val images')
    cls = np.load(CLS_LABELS, allow_pickle=True).item()

    os.makedirs('analysis', exist_ok=True)
    out_lines = []
    def out(s=''):
        print(s)
        out_lines.append(s)

    # ---- per-image mIoU for all configs + sanity check on global mIoU ----
    miou = {}            # config -> per-image array
    gt_cache = {}
    for d, t in [(BASELINE_DIR, T_BASE)] + [(d, T_RUN) for pair in cam_pairs for d in pair]:
        per = np.full(len(names), np.nan)
        Pg = np.zeros(21); Tg = np.zeros(21); TPg = np.zeros(21)
        for i, name in enumerate(tqdm(names, desc=d)):
            cam = np.load(os.path.join(d, name + '.npy'), allow_pickle=True).item()
            if name not in gt_cache:
                gt_cache[name] = np.array(Image.open(os.path.join(GT, name + '.png')))
            gt = gt_cache[name]
            per[i] = per_image_miou(cam, gt, t)
            # global accumulation
            h, w = list(cam.values())[0].shape
            fg = np.zeros((20, h, w), np.float32)
            for k in cam:
                fg[k] = cam[k]
            pred = np.where(fg.max(axis=0) > t, fg.argmax(axis=0) + 1, 0).astype(np.uint8)
            cal = gt < 255
            gt_c = gt[cal].astype(np.int64); p_c = pred[cal].astype(np.int64)
            Pg += np.bincount(p_c, minlength=21)
            Tg += np.bincount(gt_c, minlength=21)
            TPg += np.bincount(gt_c[p_c == gt_c], minlength=21)
        miou[d] = per
        gm = np.mean(TPg / (Tg + Pg - TPg + 1e-10)) * 100
        exp = EXPECTED_MIOU.get(d)
        tag = f'  (expected {exp})' if exp else ''
        flag = 'OK' if exp is None or abs(gm - exp) < 0.005 else 'MISMATCH!'
        out(f'{d:28s} global mIoU = {gm:6.2f}%  {tag}  [{flag}]')

    # ---- per-image damage (paired, mean over 3 seeds) ----
    damage = np.zeros(len(names))
    for only_d, spfr_d in cam_pairs:
        damage += (miou[spfr_d] - miou[only_d]) / len(cam_pairs)

    # ---- co-occurrence groups from cls_labels ----
    n_fg = np.zeros(len(names), dtype=int)
    class_set = [None] * len(names)
    for i, name in enumerate(names):
        lab = np.asarray(cls[name]).reshape(-1)
        present = [categories[j + 1] for j in np.where(lab > 0.5)[0]]
        class_set[i] = present
        n_fg[i] = len(present)

    single = damage[n_fg == 1]
    multi = damage[n_fg >= 2]

    def perm_pvalue(a, b, n_perm=20000, seed=0):
        rng = np.random.RandomState(seed)
        x = np.concatenate([a, b]); n = len(a)
        obs = abs(a.mean() - b.mean())
        cnt = 0
        for _ in range(n_perm):
            rng.shuffle(x)
            if abs(x[:n].mean() - x[n:].mean()) >= obs:
                cnt += 1
        return obs, (cnt + 1) / (n_perm + 1)

    out('')
    out('=' * 78)
    out('  Per-image damage  (mIoU delta = SPFR - FT-only, mean over 3 seed pairs)')
    out('=' * 78)
    out(f'  all images (n={len(names)}):   mean {damage.mean():+.2f}%  std {damage.std():.2f}%'
        f'  hurt(<-1%) {100 * (damage < -1).mean():.1f}%  improved(>+1%) {100 * (damage > 1).mean():.1f}%')
    out('')
    out(f'  single-object images (n={len(single)}): mean {single.mean():+.2f}% +/- {single.std():.2f}%')
    out(f'  multi-object images  (n={len(multi)}):  mean {multi.mean():+.2f}% +/- {multi.std():.2f}%')
    obs, p = perm_pvalue(single, multi)
    out(f'  difference = {multi.mean() - single.mean():+.2f}%  (permutation test: p = {p:.4f}, n_perm=20000)')

    out('')
    out(f'{"group":34s} {"n":>5s} {"mean dmg":>9s} {"std":>7s} {"p-val":>7s}')
    out('-' * 66)

    def report_group(label, mask):
        g = damage[mask]
        if len(g) < 10:
            return
        rest = damage[~mask]
        obs, p = perm_pvalue(g, rest) if len(rest) >= 10 else (np.nan, np.nan)
        out(f'{label:34s} {len(g):5d} {g.mean():+8.2f}% {g.std():7.2f}% {p:7.4f}')

    # person groups
    person_mask = np.array([('person' in s) for s in class_set])
    person_only = person_mask & (n_fg == 1)
    person_multi = person_mask & (n_fg >= 2)
    report_group('person alone (n_fg==1)', person_only)
    report_group('person + other classes (n_fg>=2)', person_multi)
    out('')

    # specific pairs
    for a, b in PAIRS:
        mask = np.array([(a in s) and (b in s) for s in class_set])
        report_group(f'{a}+{b} co-occurrence', mask)

    # ---- top damaged / improved images ----
    order = np.argsort(damage)
    out('')
    out('  Top-20 most damaged images (SPFR hurts):')
    for i in order[:20]:
        out(f'    {names[i]:15s} dmg={damage[i]:+7.2f}%  n_fg={n_fg[i]}  classes={"+".join(class_set[i])}')
    out('  Top-20 most improved images (SPFR helps):')
    for i in order[-1:-21:-1]:
        out(f'    {names[i]:15s} dmg={damage[i]:+7.2f}%  n_fg={n_fg[i]}  classes={"+".join(class_set[i])}')

    with open('analysis/case_damage_analysis.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines) + '\n')

    np.savez('analysis/per_image_stats.npz',
             names=np.array(names), damage=damage, n_fg=n_fg,
             miou_only_s1=miou[cam_pairs[0][0]], miou_spfr_s1=miou[cam_pairs[0][1]],
             miou_only_s2=miou[cam_pairs[1][0]], miou_spfr_s2=miou[cam_pairs[1][1]],
             miou_only_s3=miou[cam_pairs[2][0]], miou_spfr_s3=miou[cam_pairs[2][1]],
             miou_baseline=miou[BASELINE_DIR])
    print('\nSaved: analysis/case_damage_analysis.txt, analysis/per_image_stats.npz')


if __name__ == '__main__':
    main()
