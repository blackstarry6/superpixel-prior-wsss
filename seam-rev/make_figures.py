"""
生成论文可视化图:
  图5: DRS 伪标签 vs CAM 阈值伪标签对比（同一张图，两种标签）
  图4: 分割结果对比（A组/B组/DRSA/DRSB 的预测并排）

用法: python make_figures.py
输出: SEAM-REV-V1/figures/fig5_pseudo_labels.png, fig4_segmentation.png
"""
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

VOC_ROOT = 'F:/ImageSegmentation/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
SEAM_ROOT = 'F:/ImageSegmentation/SEAM-REV-V1'
SEG_ROOT = 'F:/ImageSegmentation/semantic-segmentation-codebase-main/semantic-segmentation-codebase-main'

# VOC 调色板（标准 VOC colormap）
def voc_colormap():
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        m = i
        cmap[i, 0] = (m & 1) << 7 | (m & 8) << 3
        cmap[i, 1] = (m & 2) << 6 | (m & 16) << 2
        cmap[i, 2] = (m & 4) << 5
    cmap[255] = [255, 255, 255]
    return cmap

CMAP = voc_colormap()

def label_to_color(label):
    """类别索引图 → 彩色图"""
    if label.ndim == 3:
        label = label[:, :, 0]
    return CMAP[label]

# 图5/图4 用 train 集的伪标签样本（伪标签只对 train_aug 生成）
SAMPLE_NAMES = ['2007_000039', '2007_000063', '2007_000121']

# 图3 必须用 val 集的图（分割预测只对 val 推理），按"DRSA 单图 mIoU − A 单图 mIoU"挑选
# 2009_004507: Δ+0.609 (0.286→0.895), 2008_005097: Δ+0.546 (0.414→0.960), 2007_003194: Δ+0.510
SAMPLE_NAMES_VAL = ['2009_004507', '2008_005097', '2007_003194']


def load_pseudo(ver, name):
    path = os.path.join(SEAM_ROOT, ver, f'{name}.png')
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path))


def load_pred(grp, name):
    path = os.path.join(SEG_ROOT, 'data/VOCdevkit/results/VOC2012/Segmentation',
                       grp, f'{name}.png')
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path))


def main():
    os.makedirs(os.path.join(SEAM_ROOT, 'figures'), exist_ok=True)

    # ── 图5: DRS 伪标签 vs CAM 阈值伪标签 ──────────────────
    fig, axes = plt.subplots(len(SAMPLE_NAMES), 3, figsize=(5.8, 1.55 * len(SAMPLE_NAMES)))
    for r, name in enumerate(SAMPLE_NAMES):
        img = np.array(Image.open(os.path.join(VOC_ROOT, 'JPEGImages', f'{name}.jpg')))
        pseudo_old = load_pseudo('pseudo_train', name)
        pseudo_new = load_pseudo('pseudo_train_drs', name)

        axes[r, 0].imshow(img); axes[r, 0].set_title(f'{name} input', fontsize=8)
        axes[r, 0].axis('off')
        if pseudo_old is not None:
            axes[r, 1].imshow(label_to_color(pseudo_old)); axes[r, 1].set_title('CAM-threshold pseudo labels', fontsize=8)
        axes[r, 1].axis('off')
        if pseudo_new is not None:
            axes[r, 2].imshow(label_to_color(pseudo_new)); axes[r, 2].set_title("DRS-refined pseudo labels", fontsize=8)
        axes[r, 2].axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(SEAM_ROOT, 'figures', 'fig5_pseudo_labels.png'), dpi=300)
    fig.savefig(os.path.join(SEAM_ROOT, 'figures', 'Fig5_pseudo_labels.png'), dpi=440)  # TVC hi-res
    plt.close()
    print('图5 已保存: figures/fig5_pseudo_labels.png')

    # ── 图3: 分割结果对比 ─────────────────────────────────
    groups = [
        ('A (CAM-threshold, CE)', 'deeplabv1_val_A组'),
        ('DRSA (DRS, CE)', 'deeplabv1_val_DRSA组'),
        ('DRSB (DRS, CE+SPFR)', 'deeplabv1_val_DRSB组'),
    ]
    fig, axes = plt.subplots(len(SAMPLE_NAMES_VAL), len(groups) + 2, figsize=(5.8, 1.30 * len(SAMPLE_NAMES_VAL)))
    for r, name in enumerate(SAMPLE_NAMES_VAL):
        img = np.array(Image.open(os.path.join(VOC_ROOT, 'JPEGImages', f'{name}.jpg')))
        gt = np.array(Image.open(os.path.join(VOC_ROOT, 'SegmentationClass', f'{name}.png')))
        gt[gt == 255] = 0  # void → 黑色

        axes[r, 0].imshow(img); axes[r, 0].set_title(f'{name} input', fontsize=8); axes[r, 0].axis('off')
        axes[r, 1].imshow(label_to_color(gt)); axes[r, 1].set_title('GT', fontsize=8); axes[r, 1].axis('off')

        for c, (label, grp) in enumerate(groups):
            pred = load_pred(grp, name)
            if pred is not None:
                axes[r, c + 2].imshow(label_to_color(pred))
            axes[r, c + 2].set_title(label, fontsize=8)
            axes[r, c + 2].axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(SEAM_ROOT, 'figures', 'fig4_segmentation.png'), dpi=300)
    fig.savefig(os.path.join(SEAM_ROOT, 'figures', 'Fig3_segmentation.png'), dpi=440)  # TVC hi-res
    plt.close()
    print('图4 已保存: figures/fig4_segmentation.png')

    # ── 图2: 伪标签背景缺陷示意（修复前后）────────────────
    # 用 2007_000032（修复前后对比最明显）
    name = '2007_000032'
    fig, axes = plt.subplots(1, 3, figsize=(5.8, 2.0))
    img = np.array(Image.open(os.path.join(VOC_ROOT, 'JPEGImages', f'{name}.jpg')))
    pseudo_bug = load_pseudo('pseudo_train_v1', name)  # 错误版本（背景0%）
    pseudo_fix = load_pseudo('pseudo_train', name)     # 修复版本

    axes[0].imshow(img); axes[0].set_title('Input', fontsize=8); axes[0].axis('off')
    if pseudo_bug is not None:
        axes[1].imshow(label_to_color(pseudo_bug)); axes[1].set_title('Defective (bg = 0%)', fontsize=8)
    axes[1].axis('off')
    if pseudo_fix is not None:
        axes[2].imshow(label_to_color(pseudo_fix)); axes[2].set_title('Fixed (bg restored)', fontsize=8)
    axes[2].axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(SEAM_ROOT, 'figures', 'fig2_pseudo_bug.png'), dpi=300)
    fig.savefig(os.path.join(SEAM_ROOT, 'figures', 'FigS2_pseudo_bug.png'), dpi=440)  # TVC ESM hi-res
    plt.close()
    print('图2 已保存: figures/fig2_pseudo_bug.png')

    print('\n全部图生成完毕!')


if __name__ == '__main__':
    main()
