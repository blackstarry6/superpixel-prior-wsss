"""FT-only s1 completion of the probe-2 leakage audit (NEW file; nothing
existing is modified or overwritten).

The audit (analyze_probe_imgsplit.py) covered the conv5 BASELINE column of
the probe table; this script completes the FT-only s1 column:
  1. extract FT-only s1 conv5 features (448 input) ->
     analysis/fm_feat/conv5_ftonly_s1/{name}.npy (float16, one-time)
  2. anchor (sample-level split, must reproduce published 85.7/67.4)
     + byimage audit (240/60 images).

Usage: python analyze_probe_imgsplit_ft.py
Output: analysis/probe2_imgsplit_ft.txt
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_topk_probe import img448_tensor, load_names
from analyze_probe_imgsplit import collect_cells, two_classifiers

FT_WEIGHTS = 'finetune_only_s1.pth'
FEAT_DIR = 'analysis/fm_feat/conv5_ftonly_s1'
PUBLISHED_FT = (85.7, 67.4)      # (linear, proto) from Table 6, FT-only s1
N = 300


def extract():
    from network.resnet38_SEAM import Net
    model = Net().cuda().eval()
    sd = torch.load(FT_WEIGHTS, map_location='cpu')
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    os.makedirs(FEAT_DIR, exist_ok=True)
    names = load_names(N)
    done = 0
    for i, name in enumerate(names):
        fp = os.path.join(FEAT_DIR, name + '.npy')
        if os.path.exists(fp):
            continue
        with torch.no_grad():
            _, _, conv5 = model(img448_tensor(name).unsqueeze(0).cuda(),
                                return_feat=True)
        np.save(fp, conv5[0].cpu().numpy().astype(np.float16))
        done += 1
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(names)}')
    print(f'extraction done ({done} new files)')


def audit():
    F, y, im = collect_cells(FEAT_DIR)

    n_train = int(0.8 * len(y))
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(y))
    Fs, ys = F[perm], y[perm]
    lin_a, pr_a = two_classifiers(Fs[:n_train], ys[:n_train],
                                  Fs[n_train:], ys[n_train:])

    imgs = np.unique(im)
    rng2 = np.random.RandomState(0)
    perm2 = rng2.permutation(len(imgs))
    tr_im = set(imgs[perm2[:int(0.8 * len(imgs))]].tolist())
    m_tr = np.isin(im, list(tr_im))
    lin_b, pr_b = two_classifiers(F[m_tr], y[m_tr], F[~m_tr], y[~m_tr])

    pl, pp = PUBLISHED_FT
    lines = ['  FT-only s1 (conv5) leakage-audit completion',
             f'  cells pooled: {len(y)} (300 val images, vf>=0.5, cap 300/img)',
             f'    sample  linear={lin_a:5.1f} proto={pr_a:5.1f}   '
             f'anchor (published {pl}/{pp}; delta {lin_a-pl:+.1f}/{pr_a-pp:+.1f})',
             f'    byimage linear={lin_b:5.1f} proto={pr_b:5.1f}   '
             f'(delta vs anchor {lin_b-lin_a:+.1f}/{pr_b-pr_a:+.1f})']
    rep = '\n'.join(lines)
    print(rep)
    with open('analysis/probe2_imgsplit_ft.txt', 'w', encoding='utf-8') as fh:
        fh.write(rep + '\n')
    print('Saved: analysis/probe2_imgsplit_ft.txt')


if __name__ == '__main__':
    extract()
    audit()
