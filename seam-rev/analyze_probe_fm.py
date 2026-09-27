"""㉓-B: Foundation-model probe replication (CLIP ViT-B/16 + DINOv2 ViT-B/14).

Runs the SAME two probes as analyze_probe.py (boundary alignment +
representation structure) on foundation-model patch features, to test
whether the premise defect (absent same-object similarity structure in raw
feature space) is CNN-specific or architecture-general.

Stages (run with --stage 0/1/2):
  0  smoke test: single image, assert feature grids (28x28 / 32x32)
  1  extraction: 300 val images -> cached npy per model (GPU, one-time)
  2  probes:     reuse cell_stats/feat_grad_mag/boundary_ratio from
                 analyze_probe.py verbatim; output analysis/probe_fm.txt

Paths are all NEW (fm_prefixed); nothing existing is overwritten:
  features : analysis/fm_feat/{clip_vitb16,dinov2_vitb14}/{imgid}.npy
  output   : analysis/probe_fm.txt
"""
import argparse
import os
import sys

import numpy as np
import torch

SEAM_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SEAM_ROOT)

SIZE = 448
N_IMAGES = 300
GT = 'VOC2012/SegmentationClass'
FEAT_DIR = {'clip_vitb16': 'analysis/fm_feat/clip_vitb16',
            'dinov2_vitb14': 'analysis/fm_feat/dinov2_vitb14'}
OUT = 'analysis/probe_fm.txt'

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IN_MEAN = (0.485, 0.456, 0.406)
IN_STD = (0.229, 0.224, 0.225)

NORM = {'clip_vitb16': (CLIP_MEAN, CLIP_STD), 'dinov2_vitb14': (IN_MEAN, IN_STD)}
GRIDS = {'clip_vitb16': 28, 'dinov2_vitb14': 32}   # 448/16, 448/14


# ----------------------------------------------------------------- models
def load_clip():
    """open_clip ViT-B/16 with pos-embed interpolated 197 -> 785 tokens."""
    import open_clip
    from torch.nn import functional as F
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    v = model.visual
    old = v.positional_embedding.data            # (197, 768)
    cls, grid = old[:1], old[1:]                 # (1,768), (196,768)
    g = grid.reshape(14, 14, -1).permute(2, 0, 1).unsqueeze(0)      # (1,768,14,14)
    g = F.interpolate(g, size=(28, 28), mode='bicubic', align_corners=False)
    g = g.squeeze(0).permute(1, 2, 0).reshape(784, -1)
    v.positional_embedding = torch.nn.Parameter(torch.cat([cls, g], 0).to(old.dtype))
    return model.eval()


def load_dinov2():
    """timm DINOv2 ViT-B/14 at img_size=448, pos-embed resampled 37x37 -> 32x32."""
    import timm
    from timm.layers import resample_abs_pos_embed
    from safetensors.torch import load_file
    sd = load_file(os.path.join(SEAM_ROOT, 'fm_weights', 'dinov2_vitb14.safetensors'))
    pe = sd.pop('pos_embed')                     # (1,1370,768)
    pe = resample_abs_pos_embed(pe, new_size=(32, 32))
    model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False,
                              num_classes=0, img_size=448)
    missing, unexpected = model.load_state_dict({**sd, 'pos_embed': pe}, strict=True), None
    return model.eval()


@torch.no_grad()
def features(model_key, model, img_f32cn):
    """img_f32cn: (1,3,448,448) normalized. Returns (C, gh, gw) patch grid."""
    if model_key == 'clip_vitb16':
        v = model.visual
        x = v.conv1(img_f32cn)                              # (1,C,28,28)
        b, c, gh, gw = x.shape
        x = x.reshape(b, c, -1).permute(0, 2, 1)            # (1,784,C)
        if v.class_embedding is not None:
            cls = v.class_embedding.to(x.dtype) + torch.zeros(1, 1, c, dtype=x.dtype, device=x.device)
            x = torch.cat([cls, x], dim=1)
        x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = v.transformer(x)
        x = x.permute(1, 0, 2)
        return x[0, 1:].permute(1, 0).reshape(gh, gw, -1).permute(2, 0, 1).cpu().numpy()
    else:  # dinov2
        f = model.forward_features(img_f32cn)               # (1,1025,768)
        gh = gw = 32
        return f[0, 1:].permute(1, 0).reshape(gh, gw, -1).permute(2, 0, 1).cpu().numpy()


def normalize(img_u8, mean, std):
    t = img_u8.astype(np.float32) / 255.
    return torch.from_numpy(((t - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32))
                            .transpose(2, 0, 1).copy()).unsqueeze(0).float()


# ----------------------------------------------------------------- stage 0
def stage0():
    from PIL import Image
    import cv2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[stage0] device={device}')
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    cv2.circle(img, (224, 224), 100, (255, 255, 255), -1)

    for key in ['clip_vitb16', 'dinov2_vitb14']:
        model = load_clip().to(device) if key == 'clip_vitb16' else load_dinov2().to(device)
        mean, std = NORM[key]
        x = normalize(img, mean, std).to(device)
        f = features(key, model, x)
        g = GRIDS[key]
        assert f.shape[1] == g and f.shape[2] == g, \
            f'{key}: grid {f.shape[1]}x{f.shape[2]} != {g}x{g}'
        assert np.isfinite(f).all(), f'{key}: non-finite features'
        print(f'[stage0] {key}: PASS  feature grid {f.shape} (expect {g}x{g}), '
              f'range [{f.min():.3f}, {f.max():.3f}]')
        del model
        torch.cuda.empty_cache()
    print('[stage0] ALL PASS')


# ----------------------------------------------------------------- stage 1
def stage1():
    """Extract + cache patch features for the first 300 val images (same list
    as analyze_probe.py). Skips images whose npy already exists."""
    from PIL import Image
    from tqdm import tqdm
    import cv2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    val_names = [l.split()[0].strip().split('/')[-1][:-4]
                 for l in open('voc12/val.txt').read().strip().split('\n')][:N_IMAGES]
    jpeg = 'VOC2012/JPEGImages'
    for key in FEAT_DIR:
        os.makedirs(FEAT_DIR[key], exist_ok=True)
    models = {'clip_vitb16': load_clip().to(device),
              'dinov2_vitb14': load_dinov2().to(device)}
    for key, model in models.items():
        mean, std = NORM[key]
        todo = [n for n in val_names if not os.path.exists(
            os.path.join(FEAT_DIR[key], n + '.npy'))]
        print(f'[stage1] {key}: {len(todo)}/{len(val_names)} to extract')
        for name in tqdm(todo, desc=key):
            img = np.array(Image.open(os.path.join(jpeg, name + '.jpg')).convert('RGB'))
            img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            x = normalize(img, mean, std).to(device)
            f = features(key, model, x).astype(np.float32)
            np.save(os.path.join(FEAT_DIR[key], name + '.npy'), f)
        del model
        torch.cuda.empty_cache()
    print('[stage1] DONE')


# ----------------------------------------------------------------- stage 2
def stage2():
    """Probes on cached features, reusing analyze_probe.py functions verbatim."""
    from analyze_probe import cell_stats, feat_grad_mag, boundary_ratio, rgb_grad_mag
    from PIL import Image
    import cv2
    from sklearn.linear_model import RidgeClassifier

    val_names = [l.split()[0].strip().split('/')[-1][:-4]
                 for l in open('voc12/val.txt').read().strip().split('\n')][:N_IMAGES]
    jpeg = 'VOC2012/JPEGImages'

    lines = []
    lines.append('=' * 72)
    lines.append('  Foundation-model probe replication (㉓-B)')
    lines.append(f'  images: {len(val_names)} val (same as analyze_probe.py), 448x448,')
    lines.append('  final-layer patch tokens; per-model own preprocessing')
    lines.append('=' * 72)

    for key in FEAT_DIR:
        b_ratios_f, b_ratios_rgb = [], []
        n_b = n_i = 0
        F_list, y_list = [], []
        for name in val_names:
            img = np.array(Image.open(os.path.join(jpeg, name + '.jpg')).convert('RGB'))
            img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            gt = np.array(Image.open(os.path.join(GT, name + '.png')))
            gt = cv2.resize(gt, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

            f = np.load(os.path.join(FEAT_DIR[key], name + '.npy'))  # (C,h,w)
            Hf, Wf = f.shape[1], f.shape[2]
            bfrac, vf, maj = cell_stats(gt, Hf, Wf)
            rg = rgb_grad_mag(img).reshape(Hf, SIZE // Hf, Wf, SIZE // Wf).mean(axis=(1, 3))
            fg = feat_grad_mag(f)
            r_f, cb, ci = boundary_ratio(fg, bfrac, vf)
            r_rgb, _, _ = boundary_ratio(rg, bfrac, vf)
            b_ratios_f.append(r_f)
            b_ratios_rgb.append(r_rgb)
            n_b += cb; n_i += ci

            mask = (vf >= 0.5)
            sel = np.argwhere(mask)
            if len(sel) > 300:
                idx = np.random.RandomState(0).choice(len(sel), 300, replace=False)
                sel = sel[idx]
            for i, j in sel:
                F_list.append(f[:, i, j])
                y_list.append(maj[i, j])

        F = np.stack(F_list).astype(np.float32)
        y = np.array(y_list)
        n_train = int(0.8 * len(y))
        rng = np.random.RandomState(0)
        perm = rng.permutation(len(y))
        F, y = F[perm], y[perm]
        Xtr, ytr, Xte, yte = F[:n_train], y[:n_train], F[n_train:], y[n_train:]

        clf = RidgeClassifier(alpha=1.0)
        clf.fit(Xtr, ytr)
        lin_acc = 100 * clf.score(Xte, yte)

        cent = np.stack([Xtr[ytr == c].mean(axis=0) for c in range(21)])
        cent = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-8)
        Xte_n = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
        proto_acc = 100 * (np.argmax(Xte_n @ cent.T, axis=1) == yte).mean()

        lines.append(f'--- {key} ---')
        lines.append(f'  cells: boundary={n_b}  interior={n_i}')
        lines.append(f'  boundary/interior gradient ratio:  feat = {np.nanmean(b_ratios_f):.2f}'
                     f'   RGB reference = {np.nanmean(b_ratios_rgb):.2f}')
        lines.append(f'  pixel classification: linear = {lin_acc:.1f}%  prototype = {proto_acc:.1f}%'
                     f'  (chance 4.8%)')
        lines.append('')

    # SEAM reference numbers copied from analysis/probe_analysis.txt (baseline row)
    lines.append('--- SEAM conv5 (reference, from analyze_probe.py, 56x56 grid) ---')
    lines.append('  boundary/interior gradient ratio:  conv5 = 0.79   RGB = 0.85')
    lines.append('  pixel classification: linear = 85.3%  prototype = 66.2%  (baseline)')
    lines.append('')
    lines.append('Interpretation:')
    lines.append('  Note: grids differ (CNN 56x56, CLIP 28x28, DINOv2 32x32); RGB references')
    lines.append('  are recomputed per grid, so cross-model ratio comparisons are grid-relative.')
    lines.append('  feat ratio below its own RGB ref + linear >> prototype => premise defect')
    lines.append('  present (features do not sharpen at semantic boundaries; classification')
    lines.append('  space, not a similarity metric).')

    report = '\n'.join(lines)
    print(report)
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(report + '\n')
    print(f'Saved: {OUT}')


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, default=0, choices=[0, 1, 2])
    args = ap.parse_args()
    if args.stage == 0:
        stage0()
    elif args.stage == 1:
        stage1()
    elif args.stage == 2:
        stage2()


if __name__ == '__main__':
    main()
