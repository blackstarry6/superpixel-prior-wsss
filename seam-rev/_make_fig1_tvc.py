"""Fig. 1 rebuild (TVC): two-panel modular design, vector PDF + 300-dpi PNG.

(a) WSSS pipeline with the two SPFR insertion points and the DRS branch
    (annotation uses the canonical 3-seed ruler +15.97%, fixing the stale
    +14.79% in the old fig1_pipeline.png).
(b) SPFR detail panel: superpixel partition -> prototype + L_intra ->
    adjacent-pair L_inter (pull/push with CAM-entropy weight).

Designed at final embed size 174 mm (6.85 in) so font points are final points.
No in-figure title (journal convention); panel letters (a)/(b) only.
Run from F:/ImageSegmentation.  Old fig1_pipeline.png is left untouched.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType in PDF (Springer: avoid Type 3)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle
import os

fig, ax = plt.subplots(figsize=(6.85, 3.62))
ax.set_xlim(0, 100)
ax.set_ylim(0, 53)
ax.axis('off')

PIPE = '#DCE9F5'   # standard pipeline   (blue)
DRS = '#DFF2DF'    # DRS branch          (green)
SPFR = '#FDE7CF'   # SPFR insertion      (orange)
PANEL = '#F7F7F9'  # panel (b) background
EC = '#3B3B3B'
GREEN, RED, BLUE = '#2E7D32', '#C0392B', '#1F5FA8'


def _rend():
    return fig.canvas.get_renderer()


def box(x, y, w, h, text, fc, ec=EC, fs=6.4, tc='black', bold=False, lw=0.8):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.10,rounding_size=0.55',
                       linewidth=lw, facecolor=fc, edgecolor=ec, mutation_aspect=0.55)
    ax.add_patch(p)
    t = ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
                color=tc, fontweight='bold' if bold else 'normal', linespacing=1.12)
    for _ in range(50):                      # measured auto-fit (no overflow possible)
        fig.canvas.draw()
        tb, pb = t.get_window_extent(_rend()), p.get_window_extent(_rend())
        if tb.width <= pb.width * 0.92 and tb.height <= pb.height * 0.86:
            break
        fs -= 0.2
        t.set_fontsize(fs)
    return t


def arrow(x1, y1, x2, y2, color=EC, lw=1.0, style='-|>', ms=7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=ms, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0))

# ══ Panel (a): pipeline (y 27–53) ════════════════════════════════
ax.text(1.0, 51.3, '(a)', fontsize=8.5, fontweight='bold', va='top')
PY, BH = 38.0, 7.2
box(1.0, PY, 8.5, BH, 'Input\nimage', '#EFEFEF', fs=6.4)
box(12.0, PY, 20.0, BH, 'Classification network\nSEAM (WRN-38 + PCM)', PIPE, fs=6.4, bold=True)
box(35.0, PY, 7.5, BH, 'CAM', PIPE, fs=6.8, bold=True)
box(45.5, PY, 12.5, BH, 'Pseudo-label\ngeneration', PIPE, fs=6.4)
box(60.5, PY, 10.5, BH, 'Pseudo-\nlabels', PIPE, fs=6.4, bold=True)
box(73.5, PY, 19.0, BH, 'Segmentation network\nDeepLabV1 (ResNet-38)', PIPE, fs=6.4, bold=True)
arrow(9.5, PY + BH / 2, 12.0, PY + BH / 2)
arrow(32.0, PY + BH / 2, 35.0, PY + BH / 2)
arrow(42.5, PY + BH / 2, 45.5, PY + BH / 2)
arrow(58.0, PY + BH / 2, 60.5, PY + BH / 2)
arrow(71.0, PY + BH / 2, 73.5, PY + BH / 2)
arrow(92.5, PY + BH / 2, 96.5, PY + BH / 2)
ax.text(97.0, PY + BH / 2 + 0.7, 'Seg.\nresult', ha='left', va='center',
        fontsize=6.2, fontweight='bold')

# DRS refinement branch (above CAM → pseudo-labels)
box(30.0, 48.2, 26.0, 4.4, 'AffinityNet + DRS random-walk refinement', DRS, fs=6.2, bold=True)
arrow(38.7, PY + BH, 38.7, 48.2, color=GREEN, lw=1.1)
arrow(56.0, 50.4, 63.5, PY + BH, color=GREEN, lw=1.1)
ax.text(64.5, 49.9, 'high-quality pseudo-labels\n(+15.97% mIoU, established ruler)',
        ha='left', va='center', fontsize=5.8, color=GREEN, style='italic')

# SPFR insertion points (below the two networks)
SY, SH = 27.6, 6.8
box(12.0, SY, 20.0, SH, 'SPFR \u2460  Exp. 1\n$L_{intra} + L_{inter}$ on conv5 (1024-d)', SPFR, fs=6.0, bold=True)
arrow(22.0, SY + SH, 22.0, PY, color=RED, lw=1.3)
box(73.5, SY, 19.0, SH, 'SPFR \u2461  Exp. 2\n$0.1\\,L_{inter}$ on conv5', SPFR, fs=6.0, bold=True)
arrow(83.0, SY + SH, 83.0, PY, color=RED, lw=1.3)

# legend (single row, direct under panel a)
lx = 2.0
for fc, lab in [(SPFR, 'SPFR insertion (no measurable gain, 3-seed paired)'),
                (DRS, 'DRS refinement (+15.97%)'),
                (PIPE, 'standard pipeline')]:
    ax.add_patch(FancyBboxPatch((lx, 23.4), 1.7, 1.5, boxstyle='round,pad=0.06',
                                facecolor=fc, edgecolor=EC, lw=0.6, mutation_aspect=0.55))
    t = ax.text(lx + 2.4, 24.15, lab, fontsize=5.6, va='center')
    fig.canvas.draw()
    x_end = ax.transData.inverted().transform(t.get_window_extent(_rend()).x1)[0] \
        if False else ax.transData.inverted().transform(
            (t.get_window_extent(_rend()).x1, 0))[0]
    lx = x_end + 3.2

# ══ Panel (b): SPFR detail (y 0–22) ═════════════════════════════
ax.add_patch(FancyBboxPatch((0.6, 0.4), 98.8, 21.2, boxstyle='round,pad=0.1,rounding_size=0.8',
                            facecolor=PANEL, edgecolor='#B9B9C2', lw=0.7, mutation_aspect=0.55))
ax.text(2.2, 20.2, '(b)  SPFR: superpixel prototype feature regularization',
        fontsize=7.0, fontweight='bold', va='center')

def blob(cx, cy, r, color, ec='#7A7A7A', alpha=1.0):
    """irregular hexagonal superpixel blob"""
    import math
    vs = [(cx + r * (0.75 + 0.28 * ((i * 7) % 3) / 2.0) * math.cos(math.pi / 3 * i + 0.35),
           cy + r * (0.80 + 0.25 * ((i * 5) % 3) / 2.0) * math.sin(math.pi / 3 * i + 0.35))
          for i in range(6)]
    ax.add_patch(Polygon(vs, closed=True, facecolor=color, edgecolor=ec, lw=0.7, alpha=alpha))

# (b1) superpixel partition
ax.text(16.5, 17.3, 'superpixel partition (SLIC)', fontsize=6.2, ha='center', style='italic')
blob(10.0, 11.5, 3.2, '#C9DCEE'); blob(16.5, 12.8, 3.4, '#F6CFA5')
blob(22.5, 10.5, 3.0, '#C9DCEE'); blob(13.0, 6.5, 3.3, '#DDDDDD')
blob(20.0, 5.5, 3.0, '#DDDDDD'); blob(7.0, 6.0, 2.8, '#DDDDDD')
ax.text(10.0, 2.6, 'superpixel $s$', fontsize=5.6, ha='center', color=BLUE)
ax.text(16.5, 16.0, '', fontsize=5)

# (b2) prototype + L_intra
ax.text(45.5, 17.3, 'prototype pooling + $L_{intra}$', fontsize=6.2, ha='center', style='italic')
blob(36.5, 11.0, 4.6, '#C9DCEE', alpha=0.35)
for (dx, dy) in [(-2.1, 1.6), (1.7, 2.0), (2.3, -1.3), (-1.5, -1.9), (0.2, 0.1)]:
    ax.add_patch(Circle((36.5 + dx, 11.0 + dy), 0.42, facecolor=BLUE, edgecolor='none', zorder=5))
    arrow(36.5 + dx, 11.0 + dy, 47.5, 12.6, color=BLUE, lw=0.7, ms=5)
ax.add_patch(Circle((47.5, 12.6), 0.85, facecolor='white', edgecolor=BLUE, lw=1.1, zorder=6))
ax.text(47.5, 12.6, 'p', fontsize=6.0, ha='center', va='center', color=BLUE, zorder=7)
ax.text(53.0, 6.2, '$p_s=$ mean of conv5 features in $s$\n'
                   '$L_{intra}=1-\\cos(f_i,\\,p_{s(i)})$',
        fontsize=5.6, ha='center', va='center')

# (b3) L_inter pull/push
ax.text(80.5, 17.3, 'adjacent pairs: $L_{inter}$', fontsize=6.2, ha='center', style='italic')
blob(68.5, 11.0, 3.6, '#C9DCEE', alpha=0.85)
blob(76.5, 11.0, 3.6, '#C9DCEE', alpha=0.85)
arrow(72.4, 11.0, 72.6, 11.0, color=GREEN, lw=1.2, style='<|-|>', ms=7)
ax.text(72.5, 15.4, '$y_s=y_t$: pull', fontsize=5.6, ha='center', color=GREEN)
blob(87.0, 6.5, 3.3, '#F6CFA5', alpha=0.85)
blob(95.2, 6.5, 3.3, '#C9DCEE', alpha=0.85)
arrow(88.6, 6.5, 85.8, 6.5, color=RED, lw=1.0, ms=6)
arrow(93.6, 6.5, 96.4, 6.5, color=RED, lw=1.0, ms=6)
ax.text(91.4, 10.2, '$y_s\\neq y_t$: push (margin)', fontsize=5.6, ha='center', color=RED)
ax.text(80.5, 2.4, 'same-class pulls weighted by $w_k = 1/(1+H_k)$  ($H_k$: CAM entropy in $s$)',
        fontsize=5.4, ha='center')

os.makedirs('SEAM-REV-V1/figures', exist_ok=True)
plt.tight_layout(pad=0.3)
os.makedirs('paper_latex_TVC/figures', exist_ok=True)
fig.savefig('paper_latex_TVC/figures/Fig1_pipeline.pdf', bbox_inches='tight')   # vector (TVC)
fig.savefig('paper_latex_TVC/figures/Fig1_pipeline.png', dpi=300, bbox_inches='tight')  # preview
print('saved Fig1_pipeline.pdf / .png')
