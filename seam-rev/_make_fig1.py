import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Designed at the final embedded size (~5.5 in wide); text auto-shrinks to fit its box
# (rendered extents are MEASURED, not estimated, so overflow is impossible).
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.set_xlim(-0.3, 22.0)
ax.set_ylim(-3.8, 4.45)
ax.axis('off')


def _renderer():
    return fig.canvas.get_renderer()


def box(x, y, w, h, text, fc, ec='#333333', fs=14, tc='black', bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                       linewidth=1.2, facecolor=fc, edgecolor=ec)
    ax.add_patch(p)
    t = ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs,
                color=tc, fontweight='bold' if bold else 'normal', linespacing=1.15)
    # auto-fit: shrink font until the rendered text fits inside the rendered box
    for _ in range(40):
        fig.canvas.draw()
        tb = t.get_window_extent(_renderer())
        pb = p.get_window_extent(_renderer())
        if tb.width <= pb.width * 0.93 and tb.height <= pb.height * 0.88:
            break
        fs -= 0.5
        t.set_fontsize(fs)
    return t


def arrow(x1, y1, x2, y2, color='#333333', lw=1.6, style='-|>'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=14, lw=lw, color=color))


PIPE = '#D6E4F0'   # light blue pipeline
DRS = '#D9F0D6'    # green DRS branch
SPFR = '#FDE2C8'   # orange SPFR insertions
IMG = '#EEEEEE'

# --- Main pipeline (y=0.2) ---
box(0.0, 0.2, 1.8, 1.0, 'Input\nimage', IMG, fs=14)
box(2.4, 0.2, 4.7, 1.0, 'Classification Net\n(SEAM, WideResNet-38\n+ PCM)', PIPE, fs=14, bold=True)
box(7.7, 0.2, 1.3, 1.0, 'CAM', PIPE, fs=15, bold=True)
box(9.6, 0.2, 3.9, 1.0, 'Pseudo-label\ngen\n(cam_to_pseudo)', PIPE, fs=14)
box(14.1, 0.2, 2.9, 1.0, 'Pseudo-\nlabels', PIPE, fs=14, bold=True)
box(17.6, 0.2, 4.0, 1.0, 'Segmentation Net\n(DeepLabV1,\nResNet-38)', PIPE, fs=14, bold=True)

# horizontal arrows on main row
arrow(1.8, 0.7, 2.4, 0.7)
arrow(7.1, 0.7, 7.7, 0.7)
arrow(9.0, 0.7, 9.6, 0.7)
arrow(13.5, 0.7, 14.1, 0.7)
arrow(17.0, 0.7, 17.6, 0.7)

# result label after segmentation
ax.text(20.9, 1.7, 'Segmentation\nresult', ha='center', va='center', fontsize=14, fontweight='bold')
arrow(20.3, 1.2, 20.8, 1.5)

# --- DRS refinement branch (y=2.2) ---
box(8.0, 2.2, 4.2, 0.9, 'AffinityNet + DRS\nrandom walk\n(refinement)', DRS, fs=14, bold=True)
arrow(8.6, 1.2, 8.6, 2.2, color='#2e7d32')            # CAM up to DRS
arrow(12.3, 2.65, 15.5, 1.2, color='#2e7d32')          # DRS -> pseudo-labels (high quality)
ax.text(15.3, 2.55, 'high-quality pseudo-labels\n(+14.79% mIoU)', ha='left', va='center',
        fontsize=12, color='#2e7d32', style='italic',
        bbox=dict(facecolor='white', edgecolor='none', pad=2.0))

# --- SPFR insertion points (below, y=-1.7) ---
box(2.4, -1.7, 4.4, 1.0, 'SPFR  \u2460  Experiment 1\nL_intra + L_inter\non conv5 (1024-d)', SPFR, fs=14, bold=True)
arrow(4.6, -0.7, 4.6, 0.2, color='#c0392b', lw=2.0)   # up into classification net

box(17.6, -1.7, 4.0, 1.0, 'SPFR  \u2461  Experiment 2\n0.1 \u00d7 L_inter\non conv5', SPFR, fs=14, bold=True)
arrow(19.6, -0.7, 19.6, 0.2, color='#c0392b', lw=2.0)  # up into segmentation net

# --- Title ---
ax.text(10.85, 4.1, 'WSSS pipeline and SPFR insertion points', ha='center', va='center',
        fontsize=19, fontweight='bold')

# --- Legend row: chips placed sequentially using MEASURED text widths; wrap to a
# --- second row if an item would cross the right margin (chips are axis-clipped).
legend_items = [(SPFR, 'SPFR insertion (no measurable gain)'),
                (DRS, 'DRS refinement (+14.79%, established)'),
                (PIPE, 'standard pipeline')]
x_cur, ly = 0.3, -3.3
for fc, label in legend_items:
    t = ax.text(x_cur + 0.45, ly + 0.15, label, fontsize=14, va='center')
    fig.canvas.draw()
    bb = t.get_window_extent(_renderer())
    x_end = ax.transData.inverted().transform((bb.x1, bb.y1))[0]
    if x_end > 21.6:                       # would overflow -> start a new row at x=0.3
        t.remove()
        ly -= 0.62
        x_cur = 0.3
        ax.add_patch(FancyBboxPatch((x_cur, ly), 0.3, 0.3, boxstyle="round,pad=0.02",
                                    facecolor=fc, edgecolor='#333'))
        t = ax.text(x_cur + 0.45, ly + 0.15, label, fontsize=14, va='center')
        fig.canvas.draw()
        bb = t.get_window_extent(_renderer())
        x_end = ax.transData.inverted().transform((bb.x1, bb.y1))[0]
        ax.set_ylim(min(ax.get_ylim()[0], ly - 0.25), ax.get_ylim()[1])
    else:
        ax.add_patch(FancyBboxPatch((x_cur, ly), 0.3, 0.3, boxstyle="round,pad=0.02",
                                    facecolor=fc, edgecolor='#333'))
    print(f'legend item "{label[:24]}..." chip@{x_cur:.2f} text_end@{x_end:.2f} row_y={ly:.2f}')
    x_cur = x_end + 0.7

plt.tight_layout()
out = 'SEAM-REV-V1/figures/fig1_pipeline.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print('saved', out)
