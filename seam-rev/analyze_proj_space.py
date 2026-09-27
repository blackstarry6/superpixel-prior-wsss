"""
H2 formal projection-space check (pre-registered supplementary verification).

Collapse detection for the 128-d projection head, per seed (s2/s3), per epoch
checkpoint:
  1. effective rank (erank) of projected pixel features
  2. dead-ReLU channel fraction
  3. superpixel-prototype cosine-similarity distribution (degenerate ~ all 1)
  4. descriptive: intra-GT-class vs inter-GT-class prototype similarity

Notes:
  - ep1 head received no SPFR gradient (warmup=1) => random-init reference.
  - final_proj == ep4_proj expected (no steps between); verified in-script.
  - Protocol mirrors analyze_probe.py (val list, 448 resize) and
    train_finetune.py (SLIC region_size=20 on ImageNet-normalized input,
    prototypes = normalized mean over feat-grid cells).
  - Prototype pairs are all-pairs over a subsample (within + across images).

Usage:  python analyze_proj_space.py [--n 100] [--seeds 2 3]
Output: analysis/proj_space_check.txt
"""
import argparse
import os
import sys

import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '.')
from network.resnet38_SEAM import Net
from spfr import generate_superpixels, resize_superpixels

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
GT_DIR = 'VOC2012/SegmentationClass'
JPEG = 'VOC2012/JPEGImages'
SIZE = 448
PIX_PER_IMG = 2000     # pixel samples per image for erank/dead stats
N_PROTO_KEEP = 4000    # prototype subsample for pairwise similarity
MIN_CELLS = 4          # min feat-grid cells for a prototype
RNG = np.random.RandomState(0)


def erank(F):
    """F: (N, D) -> effective rank = exp(entropy of normalized eigenvalues)."""
    C = np.cov(F.T)
    ev = np.linalg.eigvalsh(C).clip(min=0)
    p = ev / ev.sum()
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def load_model(path):
    m = Net().cuda().eval()
    sd = torch.load(path, map_location='cpu')
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict(sd)
    return m


def load_head(path):
    h = nn.Sequential(nn.Conv2d(1024, 128, kernel_size=1),
                      nn.ReLU(inplace=True)).cuda().eval()
    h.load_state_dict(torch.load(path, map_location='cpu'))
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seeds', type=int, nargs='+', default=[2, 3])
    args = ap.parse_args()

    val_names = [l.split(' ')[0].strip().split('/')[-1][:-4]
                 for l in open('voc12/val.txt').read().strip().split('\n')]
    names = val_names[:args.n]

    out_lines = []

    def out(s=''):
        print(s)
        out_lines.append(s)

    out('=' * 70)
    out('  H2 formal projection-space check (collapse detection)')
    out('=' * 70)
    out(f'  images: {len(names)} (val, first-n), 448x448; heads: ep1/ep2/ep3/ep4')
    out('  ep1 = random-init reference (warmup=1 => no SPFR gradient in epoch 1)')
    out(f'  pixel samples/img: {PIX_PER_IMG}; prototype subsample: {N_PROTO_KEEP}')
    out('')

    for seed in args.seeds:
        model = load_model(f'finetune_spfr_proj_s{seed}.pth')
        ckpts = [(f'ep{e}', f'finetune_spfr_proj_s{seed}_ep{e}_proj.pth')
                 for e in (1, 2, 3, 4)]
        heads = {tag: load_head(p) for tag, p in ckpts}
        sd4 = torch.load(ckpts[-1][1], map_location='cpu')
        sdf = torch.load(f'finetune_spfr_proj_s{seed}_proj.pth', map_location='cpu')
        same = all(torch.equal(sd4[k], sdf[k]) for k in sd4)
        out(f'--- seed {seed} (model: finetune_spfr_proj_s{seed}.pth; '
            f'final==ep4: {same}) ---')

        pix = {tag: [] for tag, _ in ckpts}
        protos = {tag: [] for tag, _ in ckpts}
        plabels = {tag: [] for tag, _ in ckpts}
        conv5_pix = []

        for ni, name in enumerate(tqdm(names, desc=f'seed{seed}')):
            img = np.array(Image.open(os.path.join(JPEG, name + '.jpg')).convert('RGB'))
            gt_full = np.array(Image.open(os.path.join(GT_DIR, name + '.png')))
            img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)

            t = img.astype(np.float32) / 255.
            t = (t - MEAN) / STD
            x = torch.from_numpy(t.transpose(2, 0, 1)).unsqueeze(0).cuda()
            with torch.no_grad():
                _, _, conv5 = model(x, return_feat=True)      # (1,1024,Hf,Wf)
            f = conv5[0]
            Hf, Wf = f.shape[1], f.shape[2]

            seg, _ = generate_superpixels(x[0].cpu(), n_segments=300,
                                          region_size=20)
            seg_ds = resize_superpixels(seg, Hf, Wf).cpu().numpy()   # (Hf,Wf)
            gtf = cv2.resize(gt_full, (Wf, Hf), interpolation=cv2.INTER_NEAREST)
            seg_ids = [s for s in np.unique(seg_ds)
                       if (seg_ds == s).sum() >= MIN_CELLS]

            flat = f.reshape(1024, -1).cpu().numpy()
            idx = RNG.choice(flat.shape[1],
                             min(PIX_PER_IMG, flat.shape[1]), replace=False)
            if ni < 30:                                        # conv5 reference
                conv5_pix.append(flat[:, idx].T)

            with torch.no_grad():
                pf = {tag: h(conv5).reshape(128, -1).cpu().numpy()
                      for tag, h in heads.items()}
            for tag, P in pf.items():
                pix[tag].append(P[:, idx].T)
                for s in seg_ids:
                    m = (seg_ds == s)
                    v = P[:, m.flatten()].mean(axis=1)
                    v = v / (np.linalg.norm(v) + 1e-8)
                    protos[tag].append(v)
                    lab = gtf[m]
                    valid = lab[lab < 255]
                    if len(valid) == 0:
                        plabels[tag].append(-1)
                        continue
                    bc = np.bincount(valid, minlength=21)
                    plabels[tag].append(int(bc.argmax())
                                        if bc.max() / len(valid) >= 0.9 else -1)

        del model, heads
        torch.cuda.empty_cache()

        er_conv5 = erank(np.concatenate(conv5_pix))
        out(f'  reference erank(conv5 raw, 1024-d) = {er_conv5:.1f}')
        out(f'  {"ckpt":5s} {"erank":>7s} {"dead%":>6s} {"simMean":>8s} '
            f'{"simStd":>7s} {"P5":>6s} {"P95":>6s} {">0.99":>6s} '
            f'{"intra":>7s} {"inter":>7s} {"sep":>6s}')
        for tag in pix:
            Fp = np.concatenate(pix[tag])                        # (N,128)
            er = erank(Fp)
            dead = 100.0 * float((Fp.max(axis=0) < 1e-6).mean())
            P = np.stack(protos[tag])
            if len(P) > N_PROTO_KEEP:
                keep = RNG.choice(len(P), N_PROTO_KEEP, replace=False)
                P, lab = P[keep], np.array(plabels[tag])[keep]
            else:
                lab = np.array(plabels[tag])
            S = np.clip(P @ P.T, -1, 1)
            iu = np.triu_indices(len(P), k=1)
            sims = S[iu]
            lok = (lab >= 0)
            lm = lok[iu[0]] & lok[iu[1]]
            same_m = lm & (lab[iu[0]] == lab[iu[1]])
            diff_m = lm & (lab[iu[0]] != lab[iu[1]])
            intra = sims[same_m].mean() if same_m.sum() > 10 else float('nan')
            inter = sims[diff_m].mean() if diff_m.sum() > 10 else float('nan')
            p5, p95 = np.percentile(sims, [5, 95])
            out(f'  {tag:5s} {er:7.1f} {dead:6.1f} {sims.mean():8.3f} '
                f'{sims.std():7.3f} {p5:6.3f} {p95:6.3f} '
                f'{100 * (sims > 0.99).mean():6.2f} '
                f'{intra:7.3f} {inter:7.3f} {intra - inter:6.3f}')
        out('')

    out('  Reading guide (pre-stated):')
    out('  - collapse signature: erank -> single digits, dead% dominant,')
    out('    sim distribution pinned near 1 (std ~0, >0.99 fraction ~1).')
    out('  - healthy: erank stays well above ~1/4 of dim (>=32 of 128),')
    out('    dead% small, sim std nonzero, intra > inter maintained.')
    os.makedirs('analysis', exist_ok=True)
    with open('analysis/proj_space_check.txt', 'w', encoding='utf-8') as fo:
        fo.write('\n'.join(out_lines) + '\n')
    print('\nSaved: analysis/proj_space_check.txt')


if __name__ == '__main__':
    main()
