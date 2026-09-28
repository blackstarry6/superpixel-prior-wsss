"""Figure 7 v2: per-class SPFR effect — left Exp1 (3-seed mean paired), right Exp2 low-q (3-seed paired mean).

Right panel now uses the ACTUAL 3-seed paired per-class means from
analysis/lowq_seg_3seed_full.npz (replacing the old single-seed data).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42  # TrueType in PDF (Springer: avoid Type 3)

# --- Exp 1 (classification stage): unchanged, from Table A1 ---
exp1 = {'bg': 0.06, 'aero': 0.41, 'bike': 0.12, 'bird': -0.06, 'boat': 0.05,
        'bottle': 0.06, 'bus': 0.20, 'car': -0.11, 'cat': -0.28, 'chair': 0.07,
        'cow': 0.01, 'table': 0.04, 'dog': 0.20, 'horse': -0.06, 'motor': 0.36,
        'person': 0.69, 'plant': -0.09, 'sheep': 0.15, 'sofa': -0.07,
        'train': -0.08, 'tv': 0.07}

# --- Exp 2 (low-quality arm): 3-seed paired mean from npz ---
d = np.load('analysis/lowq_seg_3seed_full.npz', allow_pickle=True)
A_dirs = ['deeplabv1_val_A组', 'deeplabv1_val_A_seed2组', 'deeplabv1_val_A_seed3组']
B_dirs = ['deeplabv1_val_B组', 'deeplabv1_val_B_seed2组', 'deeplabv1_val_B_seed3组']
A = np.stack([d[k] for k in A_dirs])  # (3, 21)
B = np.stack([d[k] for k in B_dirs])
exp2_arr = B.mean(axis=0) - A.mean(axis=0)  # (21,) paired mean

cats = ['bg', 'aero', 'bike', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'table', 'dog', 'horse', 'motor', 'person', 'plant', 'sheep', 'sofa',
        'train', 'tv']
exp2 = {c: float(v) for c, v in zip(cats, exp2_arr)}

# 3/3 consistent classes for edging
cons_neg = [c for i, c in enumerate(cats) if all(B[s][i] - A[s][i] < 0 for s in range(3))]

POS, NEG = '#009E73', '#D55E00'

fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.36))
panels = [
    (axes[0], exp1, 'Exp. 1: classification stage\n(3-seed mean of paired \u0394IoU)', False, []),
    (axes[1], exp2, 'Exp. 2: low-quality arm\n(3-seed paired mean \u0394IoU)', True, cons_neg),
]
for ax, d_dict, title, show_neg2, neg_classes in panels:
    items = sorted(d_dict.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [POS if v >= 0 else NEG for v in vals]
    edges = ['0.15' if n in neg_classes else 'none' for n in names]
    ax.barh(range(len(vals)), vals, color=colors, edgecolor=edges,
            linewidth=1.4 if neg_classes else 0, height=0.68, zorder=3)
    ax.axvline(0, color='0.25', linewidth=0.9, zorder=4)
    if show_neg2:
        ax.axvline(-1.0, color='0.45', linewidth=0.8, linestyle=':', zorder=4)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=7)
    for yi, v in enumerate(vals):
        ax.text(v + (0.05 if v >= 0 else -0.05), yi, f'{v:+.2f}',
                va='center', ha='left' if v >= 0 else 'right', fontsize=6.8,
                color='0.25')
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel('\u0394IoU (%)  (SPFR \u2212 control)', fontsize=8)
    ax.grid(axis='x', color='0.88', linewidth=0.6, zorder=0)
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.18
    ax.set_xlim(lo - pad, hi + pad)

fig.tight_layout()
fig.savefig('figures/fig7_class_diffuse.png', dpi=300, bbox_inches='tight')
fig.savefig('figures/Fig4_class_diffuse.pdf', bbox_inches='tight')  # TVC vector
fig.savefig('figures/Fig4_class_diffuse.png', dpi=440, bbox_inches='tight')  # md preview
print('saved figures/fig7_class_diffuse.png')
print(f'right panel: {len(cons_neg)} classes 3/3 negative: {cons_neg}')
