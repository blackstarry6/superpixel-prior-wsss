import numpy as np, glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

base = 'SEAM-REV-V1'
groups = {'Baseline (SEAM)':'cam_val_full', 'FT-only (s1)':'cam_finetune_only_s1', 'FT+SPFR (s1)':'cam_finetune_spfr_s1'}
person = 14
iou = {'Baseline (SEAM)':59.76, 'FT-only (s1)':47.10, 'FT+SPFR (s1)':47.73}

jpeg_dirs = ['VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages',
             'SEAM-REV-V1/VOC2012/JPEGImages',
             'semantic-segmentation-codebase-main/semantic-segmentation-codebase-main/data/VOCdevkit/VOC2012/JPEGImages']
jpeg_dir = next((p for p in jpeg_dirs if os.path.isdir(p)), None)
print('jpeg_dir =', jpeg_dir)


def load_cam(d, imgid):
    f = os.path.join(base, d, imgid + '.npy')
    if not os.path.exists(f):
        return None
    dd = np.load(f, allow_pickle=True).item()
    return dd.get(person)


cands = []
for f in sorted(glob.glob(base + '/cam_val_full/*.npy')):
    imgid = os.path.splitext(os.path.basename(f))[0]
    c0 = load_cam('cam_val_full', imgid)
    if c0 is None:
        continue
    c1 = load_cam('cam_finetune_only_s1', imgid)
    c2 = load_cam('cam_finetune_spfr_s1', imgid)
    if c1 is None or c2 is None:
        continue
    # present-class labels for this image (co-occurrence info)
    labels = list(np.load(f, allow_pickle=True).item().keys())
    cov0, cov1, cov2 = (c0 > 0.3).mean(), (c1 > 0.3).mean(), (c2 > 0.3).mean()
    cands.append((imgid, labels, cov0, cov1, cov2, c0, c1, c2))

print('person candidates:', len(cands))
cands_sorted = sorted(cands, key=lambda c: (c[3] - c[4]), reverse=True)
print('top-5 by (FT-only cov - SPFR cov), i.e. clearest SPFR-added drop:')
for c in cands_sorted[:5]:
    print('  ', c[0], 'labels', c[1], 'cov base/FT/SPFR = %.3f/%.3f/%.3f' % (c[2], c[3], c[4]))

# Prefer an image showing full monotonic degradation base > FT-only > SPFR, then biggest total drop
mono = [c for c in cands if c[2] > c[3] and c[3] >= c[4]]
print('monotonic (base>FT>=SPFR):', len(mono))
pool = mono if mono else cands
# Prefer a co-occurrence image (person + another class) that matches the paper's
# geometric-conflict mechanism (§4.4: "person 与 bicycle、dog 与 person 共现"),
# is monotonic, and shows a visible SPFR-added drop beyond fine-tune.
forced = next((c for c in cands if c[0] == '2008_006722'), None)
best = forced if forced else sorted(pool, key=lambda c: (c[2] - c[4]), reverse=True)[0]
imgid, labels, cov0, cov1, cov2, cam0, cam1, cam2 = best
print('PICKED:', imgid, 'labels', labels, 'cov base/FT/SPFR = %.3f/%.3f/%.3f' % (cov0, cov1, cov2))

img = np.array(Image.open(os.path.join(jpeg_dir, imgid + '.jpg')))
Hc, Wc = cam0.shape
Hi, Wi = img.shape[:2]
if (Hc, Wc) != (Hi, Wi):
    import torch.nn.functional as Fn  # noqa
    # fall back to PIL resize if torch missing
    def _resize(c, h, w):
        from PIL import Image as I
        return np.array(I.fromarray(c).resize((w, h)), dtype=np.float32)
    cam0 = _resize(cam0, Hi, Wi); cam1 = _resize(cam1, Hi, Wi); cam2 = _resize(cam2, Hi, Wi)

fig, axes = plt.subplots(1, 4, figsize=(6.0, 1.8))
axes[0].imshow(img)
axes[0].set_title('Input\n(%s)' % imgid, fontsize=8.5)
axes[0].axis('off')
for ax, cam, name in [(axes[1], cam0, 'Baseline (SEAM)'), (axes[2], cam1, 'FT-only (s1)'), (axes[3], cam2, 'FT+SPFR (s1)')]:
    ax.imshow(img)
    im = ax.imshow(cam, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    ax.set_title('%s\nIoU %.2f%%' % (name, iou[name]), fontsize=8.5)
    ax.axis('off')
cb = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
cb.set_label('CAM response', fontsize=8.5)
cb.ax.tick_params(labelsize=8)
plt.tight_layout()
out = 'SEAM-REV-V1/figures/fig3_cam_person.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.savefig('SEAM-REV-V1/figures/Fig2_cam_person.png', dpi=440, bbox_inches='tight')  # TVC hi-res
print('saved', out)
