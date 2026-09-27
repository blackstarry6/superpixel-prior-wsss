"""
Same-protocol evaluation: baseline + 2 old checkpoints re-inferred with
infer_epoch.py (3 scales), evaluated with the identical single-pass
threshold scan as eval_seeds.py.

Usage: python eval_same_protocol.py
"""
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

GT = 'VOC2012/SegmentationClass'
cam_dirs = {
    'Baseline(new协议)':   './cam_baseline_new',
    'FT-only 旧run(同协议)': './cam_finetune_only_old',
    'FT+SPFR 旧run(同协议)': './cam_finetune_spfr_old',
}
THRESHOLDS = [i / 100 for i in range(10, 61, 5)]
PERSON_IDX = 15  # VOC 1-20 编号中 person=15；此处 per_class 下标 0=bg,1..20 → person 在 [15]


def evaluate_dir(cam_dir):
    name_list = sorted(f[:-4] for f in os.listdir(cam_dir) if f.endswith('.npy'))
    nt = len(THRESHOLDS)
    P = np.zeros((nt, 21)); T = np.zeros((nt, 21)); TP = np.zeros((nt, 21))
    for name in tqdm(name_list, desc=os.path.basename(cam_dir)):
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
        results[label] = {'miou': r[0], 'best_t': r[1], 'per_class': r[2]}
        print(f'{label}: mIoU={r[0]:.2f}% (best_t={r[1]:.2f})')

    only = results.get('FT-only 旧run(同协议)')
    spfr = results.get('FT+SPFR 旧run(同协议)')
    base = results.get('Baseline(new协议)')
    if only and spfr:
        print(f'\n旧 run 同协议对比：')
        print(f'  Δ mIoU (FT+SPFR − FT-only) = {spfr["miou"] - only["miou"]:+.2f}%')
        print(f'  person: FT-only={only["per_class"][PERSON_IDX]:.2f} → FT+SPFR={spfr["per_class"][PERSON_IDX]:.2f}, Δ={spfr["per_class"][PERSON_IDX] - only["per_class"][PERSON_IDX]:+.2f}%')
    if base and only:
        print(f'  fine-tune 效应 (FT-only − 新协议基线) = {only["miou"] - base["miou"]:+.2f}%')


if __name__ == '__main__':
    main()
