"""Evaluate CAM mIoU for all epochs + baseline vs original SEAM."""
import numpy as np, os
from PIL import Image
from tqdm import tqdm

GT = 'VOC2012/SegmentationClass'
cam_dirs = {
    'Baseline (SEAM原文)': './cam_val_full',
    'Epoch 1 (warmup)':    './cam_ep1',
    'Epoch 2 (+SPFR)':     './cam_ep2',
    'Epoch 3 (+SPFR)':     './cam_ep3',
    'Epoch 4 (+SPFR)':     './cam_ep4',
}
categories = ['bg','aero','bike','bird','boat','bottle','bus','car','cat','chair',
              'cow','table','dog','horse','motor','person','plant','sheep','sofa','train','tv']

results = {}
for label, cam_dir in cam_dirs.items():
    if not os.path.isdir(cam_dir):
        print(f'{label}: dir not found, skip')
        continue
    
    name_list = [f[:-4] for f in os.listdir(cam_dir) if f.endswith('.npy')]
    
    # Find best background threshold
    best_miou = 0; best_t = 0
    for tt in [i/100 for i in range(10, 61, 5)]:
        TP = np.zeros(21); P = np.zeros(21); T = np.zeros(21)
        for name in name_list:
            d = np.load(os.path.join(cam_dir, name + '.npy'), allow_pickle=True).item()
            h, w = list(d.values())[0].shape
            t = np.zeros((21, h, w), np.float32); t[0] = tt
            for k in d: t[k + 1] = d[k]
            pred = np.argmax(t, axis=0).astype(np.uint8)
            gt = np.array(Image.open(os.path.join(GT, name + '.png')))
            cal = gt < 255; mask = (pred == gt) & cal
            for i in range(21):
                P[i] += ((pred == i) & cal).sum()
                T[i] += ((gt == i) & cal).sum()
                TP[i] += ((gt == i) & mask).sum()
        miou = np.mean([TP[i] / (T[i] + P[i] - TP[i] + 1e-10) for i in range(21)]) * 100
        if miou > best_miou: best_miou = miou; best_t = tt
    
    # Full per-class at best threshold
    TP = np.zeros(21); P = np.zeros(21); T = np.zeros(21)
    for name in tqdm(name_list, desc=label[:20]):
        d = np.load(os.path.join(cam_dir, name + '.npy'), allow_pickle=True).item()
        h, w = list(d.values())[0].shape
        t = np.zeros((21, h, w), np.float32); t[0] = best_t
        for k in d: t[k + 1] = d[k]
        pred = np.argmax(t, axis=0).astype(np.uint8)
        gt = np.array(Image.open(os.path.join(GT, name + '.png')))
        cal = gt < 255; mask = (pred == gt) & cal
        for i in range(21):
            P[i] += ((pred == i) & cal).sum()
            T[i] += ((gt == i) & cal).sum()
            TP[i] += ((gt == i) & mask).sum()
    
    per_class = []
    for i in range(21):
        iou = TP[i] / (T[i] + P[i] - TP[i] + 1e-10); per_class.append(iou * 100)
    miou = np.mean(per_class)
    
    results[label] = {'miou': miou, 'best_t': best_t, 'per_class': per_class}
    print(f'{label:25s}: mIoU={miou:.2f}% (best_t={best_t:.2f})')

# Summary table
print()
print('=' * 65)
print('  FINAL RESULTS — CAM mIoU (raw, no CRF, best threshold)')
print('=' * 65)
print(f'{"Model":25s} {"mIoU":>7s} {"best_t":>7s} {"vs Baseline":>11s}')
baseline_miou = results.get('Baseline (SEAM原文)', {}).get('miou', 51.23)
for label, r in results.items():
    delta = r['miou'] - baseline_miou
    marker = '✅ +' if delta > 0.5 else ('⚠️ -' if delta < -0.5 else ' ~ ')
    print(f'{label:25s} {r["miou"]:6.2f}% {r["best_t"]:6.2f}  {marker}{delta:+.2f}%')

# Per-class table for key epochs
print()
print('=' * 65)
print('  Per-Class mIoU (best threshold)')
print('=' * 65)
header = f'{"Class":10s}'
for label in results:
    header += f' {label[:12]:>8s}'
print(header)
print('-' * 65)
for i in range(21):
    row = f'{categories[i]:10s}'
    for label in results:
        row += f' {results[label]["per_class"][i]:7.2f}%'
    print(row)
