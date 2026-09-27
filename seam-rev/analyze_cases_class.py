"""
Class-restricted case test for the geometric-conflict hypothesis (Exp. 1).

Precise prediction of the binary-prototype (geometric conflict) hypothesis:
  images where a class CO-OCCURS with other foreground classes should show
  LARGER SPFR damage (more negative delta) on that class's IoU than images
  where the class appears ALONE.

Tests per-image per-class IoU delta (SPFR - FT-only, 3-seed paired mean)
for person and cat, split by co-occurrence, with permutation tests.

Sanity: image-level mean of class IoU deltas reproduces class-level net
effects from the archive (person +0.69%, cat -0.28%).

Outputs: analysis/case_class_analysis.txt

Usage: python analyze_cases_class.py
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
T_RUN = 0.15

categories = ['bg', 'aero', 'bike', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'table', 'dog', 'horse', 'motor', 'person', 'plant', 'sheep', 'sofa', 'train', 'tv']


def per_image_class_iou(cam, gt, t, cls_idx):
    """IoU of one class on one image."""
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
    T = int((gt_c == cls_idx).sum())
    P = int((p_c == cls_idx).sum())
    TP = int(((p_c == cls_idx) & (gt_c == cls_idx)).sum())
    return T, P, TP


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


def main():
    names = sorted(f[:-4] for f in os.listdir(cam_pairs[0][0]) if f.endswith('.npy'))
    cls = np.load(CLS_LABELS, allow_pickle=True).item()

    def class_delta(cls_name, present_only=True):
        idx = categories.index(cls_name)
        gt_cache = {}
        deltas = []
        n_fg_arr = []
        sets = []
        used = []
        for name in tqdm(names, desc=cls_name):
            lab = np.asarray(cls[name]).reshape(-1)
            if present_only and lab[idx - 1] < 0.5:
                continue
            if name not in gt_cache:
                gt_cache[name] = np.array(Image.open(os.path.join(GT, name + '.png')))
            gt = gt_cache[name]
            d = np.zeros(3)
            for si, (only_d, spfr_d) in enumerate(cam_pairs):
                cam_o = np.load(os.path.join(only_d, name + '.npy'), allow_pickle=True).item()
                cam_s = np.load(os.path.join(spfr_d, name + '.npy'), allow_pickle=True).item()
                To, Po, TPo = per_image_class_iou(cam_o, gt, T_RUN, idx)
                Ts, Ps, TPs = per_image_class_iou(cam_s, gt, T_RUN, idx)
                iou_o = TPo / (To + Po - TPo + 1e-10) * 100
                iou_s = TPs / (Ts + Ps - TPs + 1e-10) * 100
                d[si] = iou_s - iou_o
            deltas.append(d.mean())
            n_fg_arr.append(int((lab > 0.5).sum()))
            sets.append([categories[j + 1] for j in np.where(lab > 0.5)[0]])
            used.append(name)
        return np.array(deltas), np.array(n_fg_arr), sets, used

    os.makedirs('analysis', exist_ok=True)
    out_lines = []
    def out(s=''):
        print(s)
        out_lines.append(s)

    out('=' * 74)
    out('  Class-restricted case test of the geometric-conflict hypothesis')
    out('=' * 74)
    for cls_name, expected_net in [('person', 0.69), ('cat', -0.28)]:
        deltas, n_fg, sets, used = class_delta(cls_name)
        out('')
        out(f'--- {cls_name} (images containing {cls_name}: n={len(deltas)}) ---')
        out(f'  overall mean per-image {cls_name}-IoU delta: {deltas.mean():+.2f}% +/- {deltas.std():.2f}%'
            f'  (class-level net effect from archive: {expected_net:+.2f}%)')
        alone = deltas[n_fg == 1]
        multi = deltas[n_fg >= 2]
        if len(alone) >= 5 and len(multi) >= 5:
            obs, p = perm_pvalue(alone, multi)
            out(f'  {cls_name} alone (n={len(alone)}):  {alone.mean():+.2f}% +/- {alone.std():.2f}%')
            out(f'  {cls_name} co-occurring (n={len(multi)}): {multi.mean():+.2f}% +/- {multi.std():.2f}%')
            out(f'  difference (co-occ - alone) = {multi.mean() - alone.mean():+.2f}%  '
                f'(hypothesis predicts < 0; permutation p = {p:.4f})')
        # split co-occurring by number of extra classes
        for k in (2, 3):
            g = deltas[n_fg == k]
            if len(g) >= 5:
                out(f'  n_fg=={k} (n={len(g)}):  {g.mean():+.2f}% +/- {g.std():.2f}%')
        # top damaged for this class
        order = np.argsort(deltas)
        out(f'  top-8 most damaged {cls_name} images:')
        for i in order[:8]:
            out(f'    {used[i]:15s} d={deltas[i]:+7.2f}%  n_fg={n_fg[i]}  classes={"+".join(sets[i])}')

    with open('analysis/case_class_analysis.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines) + '\n')
    print('\nSaved: analysis/case_class_analysis.txt')


if __name__ == '__main__':
    main()
