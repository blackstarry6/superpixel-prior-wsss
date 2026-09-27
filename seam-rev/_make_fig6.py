"""Figure 6: SLIC purity distributions across the 2x2 recipe x split matrix.

Dual panel (val | train_aug), two lines per panel (rs20 vs rs25),
log-scaled y (share of superpixels per purity bucket).
Data: analysis/slic_purity_matrix_{val,train_aug}_rs{20,25}.txt (8.15M superpixels).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

buckets = ['<0.5', '0.5\u20130.7', '0.7\u20130.8', '0.8\u20130.9', '0.9\u20130.95', '\u22650.95']
data = {
    'val': {'rs20': [0.0, 1.8, 1.1, 1.7, 1.4, 93.9], 'rs25': [0.0, 2.3, 1.5, 2.3, 1.9, 92.0]},
    'train_aug': {'rs20': [0.1, 2.8, 2.0, 3.3, 2.7, 89.1], 'rs25': [0.1, 3.3, 2.4, 4.0, 3.5, 86.8]},
}
cross = {('val', 'rs20'): 2.9, ('val', 'rs25'): 3.8,
         ('train_aug', 'rs20'): 4.8, ('train_aug', 'rs25'): 5.7}
titles = {'val': 'val (1,449 images)', 'train_aug': 'train_aug (10,582 images)'}
FLOOR = 0.05

fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.45), sharey=True)
legend_handles = []
for ax, split in zip(axes, ['val', 'train_aug']):
    # shaded crossing region (purity < 0.8 = first 3 buckets)
    ax.set_xlim(-1.05, 6.1)
    ax.axvspan(-1.0, 2.5, color='0.90', zorder=0)
    x = range(6)
    for rs, color, marker, ls in [('rs20', '#0072B2', 'o', '-'),
                                  ('rs25', '#D55E00', 's', '--')]:
        ys = [max(v, FLOOR) for v in data[split][rs]]
        line = ax.plot(x, ys, marker=marker, linestyle=ls, color=color, markersize=5,
                       linewidth=1.6, label=rs, zorder=3)
        if ax is axes[0]:
            legend_handles.append(line[0])
        for xi, (yv, raw) in enumerate(zip(ys, data[split][rs])):
            lab = '<0.1' if raw < 0.1 else f'{raw:g}'
            if xi == 0:      # floor bucket: labels go LEFT of the marker (empty zone)
                off, ha = (-6, -3 if marker == 'o' else 8), 'right'
            elif xi == 5:    # top bucket: rs25 above, rs20 below-right of its marker
                off, ha = (0, 8) if marker == 's' else (7, -3), ('center' if marker == 's' else 'left')
            else:
                off, ha = (4, 7 if marker == 's' else -13), ('right' if marker == 's' else 'left')
            ax.annotate(lab, (xi, yv), textcoords='offset points',
                        xytext=off, ha=ha, fontsize=7.5, color=color)
    ax.text(1.0, 0.96, 'crossing region\n(purity < 0.8)', ha='center', va='top',
            transform=ax.get_xaxis_transform(), fontsize=8, color='0.35')
    ax.set_yscale('log')
    ax.set_ylim(FLOOR * 0.42, 700)
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(['0.1', '1', '10', '100'])
    ax.set_xticks(list(x))
    ax.set_xticklabels(buckets, fontsize=8, rotation=35, ha='right')
    ax.set_title(titles[split], fontsize=10, pad=6)
    ax.set_xlabel('Purity bucket', fontsize=9, labelpad=3)
    ax.grid(axis='y', which='major', color='0.85', linewidth=0.6, zorder=0)
axes[0].set_ylabel('Share of superpixels (%)', fontsize=9)
fig.legend(legend_handles, [h.get_label() for h in legend_handles],
           loc='lower center', bbox_to_anchor=(0.5, 1.0),
           ncol=2, fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig('figures/fig6_purity_hist.png', dpi=300, bbox_inches='tight')
fig.savefig('figures/FigS1_purity_hist.pdf', bbox_inches='tight')  # TVC ESM vector
fig.savefig('figures/FigS1_purity_hist.png', dpi=440, bbox_inches='tight')  # md preview
print('saved figures/fig6_purity_hist.png')
