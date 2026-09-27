"""
分割网络预测结果评估脚本（单进程，兼容 Windows）
=================================================
评估 test.py 输出的预测 PNG 与 GT 的 mIoU。

用法:
    python eval_seg.py                                  # 评估当前 deeplabv1_val/
    python eval_seg.py --predict_dir X --label Y        # 评估指定目录并加标签
    python eval_seg.py --compare                        # 对比所有已命名的结果目录

说明:
    - 官方 do_python_eval() 在 Windows 上因 multiprocessing 崩溃，本脚本用单进程实现相同逻辑
    - 预测目录命名约定: deeplabv1_val / deeplabv1_val_A组 / deeplabv1_val_B组 等
"""

import numpy as np
import os
import argparse
from PIL import Image

GT_DIR = 'data/VOCdevkit/VOC2012/SegmentationClass'
RESULTS_BASE = 'data/VOCdevkit/results/VOC2012/Segmentation'

CATEGORIES = ['bg', 'aero', 'bike', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'table', 'dog', 'horse', 'motor', 'person', 'plant', 'sheep', 'sofa',
              'train', 'tv']


def evaluate_dir(predict_dir, gt_dir=GT_DIR):
    """
    评估一个预测目录的 mIoU。

    Args:
        predict_dir: 预测 PNG 目录
        gt_dir: GT 标注目录

    Returns:
        (per_class_list, mIoU)
    """
    if not os.path.isdir(predict_dir):
        raise FileNotFoundError(f'预测目录不存在: {predict_dir}')

    pred_files = [f for f in os.listdir(predict_dir) if f.endswith('.png')]
    if len(pred_files) == 0:
        raise ValueError(f'预测目录为空: {predict_dir}')

    TP = np.zeros(21)
    P = np.zeros(21)
    T = np.zeros(21)

    for pf in pred_files:
        img_name = pf[:-4]
        pred = np.array(Image.open(os.path.join(predict_dir, pf)))
        gt_path = os.path.join(gt_dir, f'{img_name}.png')
        if not os.path.exists(gt_path):
            continue
        gt = np.array(Image.open(gt_path))
        cal = gt < 255                     # 排除 void 像素
        mask = (pred == gt) & cal          # 预测正确且非 void
        for i in range(21):
            P[i] += ((pred == i) & cal).sum()
            T[i] += ((gt == i) & cal).sum()
            TP[i] += ((gt == i) & mask).sum()

    per_class = [TP[i] / (T[i] + P[i] - TP[i] + 1e-10) * 100 for i in range(21)]
    miou = float(np.mean(per_class))
    return per_class, miou


def print_result(label, per_class, miou):
    """打印单个目录的评估结果"""
    print(f'\n{"=" * 55}')
    print(f'  {label}')
    print(f'{"=" * 55}')
    for i in range(21):
        print(f'  {CATEGORIES[i]:10s}: {per_class[i]:6.2f}%')
    print(f'  {"mIoU":10s}: {miou:6.2f}%')
    print(f'{"=" * 55}')


def main():
    parser = argparse.ArgumentParser(description='分割预测评估（单进程）')
    parser.add_argument('--predict_dir', type=str, default=None,
                        help='预测目录（默认 deeplabv1_val）')
    parser.add_argument('--label', type=str, default='结果',
                        help='结果标签（显示用）')
    parser.add_argument('--compare', action='store_true',
                        help='对比所有已命名的结果目录')
    parser.add_argument('--results_base', type=str, default=RESULTS_BASE,
                        help='结果根目录')
    parser.add_argument('--gt_dir', type=str, default=GT_DIR,
                        help='GT 标注目录')
    args = parser.parse_args()

    if args.compare:
        # 对比模式：列出结果根目录下所有子目录并逐一评估
        print(f'对比模式: {args.results_base}')
        dirs = sorted([d for d in os.listdir(args.results_base)
                       if os.path.isdir(os.path.join(args.results_base, d))])
        if len(dirs) == 0:
            print('  没有找到结果目录')
            return

        results = {}
        for d in dirs:
            full = os.path.join(args.results_base, d)
            try:
                per_class, miou = evaluate_dir(full, args.gt_dir)
                results[d] = (per_class, miou)
                print(f'  {d:30s}: mIoU = {miou:6.2f}%')
            except Exception as e:
                print(f'  {d:30s}: 评估失败 ({e})')

        # 汇总对比表
        print(f'\n{"=" * 70}')
        print(f'{"Class":10s}', end='')
        for d in dirs:
            if d in results:
                print(f' {d[:12]:>12s}', end='')
        print()
        print('-' * 70)
        for i in range(21):
            print(f'{CATEGORIES[i]:10s}', end='')
            for d in dirs:
                if d in results:
                    print(f' {results[d][0][i]:11.2f}%', end='')
            print()
        print('-' * 70)
        print(f'{"mIoU":10s}', end='')
        for d in dirs:
            if d in results:
                print(f' {results[d][1]:11.2f}%', end='')
        print()
        print('=' * 70)
    else:
        # 单目录模式
        predict_dir = args.predict_dir or os.path.join(args.results_base, 'deeplabv1_val')
        per_class, miou = evaluate_dir(predict_dir, args.gt_dir)
        print_result(args.label, per_class, miou)


if __name__ == '__main__':
    main()
