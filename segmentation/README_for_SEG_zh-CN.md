# 分割阶段(实验 2)——运行说明

[English](./README_for_SEG.md) | 简体中文

DeepLabV1(ResNet-38 骨干,由 SEAM 分类权重初始化)在分类阶段产出的伪标签上
训练,并在分割插入点可选地施加 SPFR 损失。

框架继承自
[semantic-segmentation-codebase](https://github.com/YudeWang/semantic-segmentation-codebase)
(上游项目见本目录 `README.md` 与 `LICENSE`);本文件说明**本研究**的实验
矩阵与运行方式。

## 实验矩阵(4 臂 × 3 种子)

| 臂 | 伪标签 | 损失 | `USE_SPFR` | `EXP_NAME` |
|---|---|---|---|---|
| DRSA | DRS 精化(高质量) | 仅 CE | `False` | `seamv1-pseudovoc-DRSA`(默认) |
| DRSB | DRS 精化(高质量) | CE + 0.1 · L_inter | `True` | `seamv1-pseudovoc-DRSB` |
| A | CAM 阈值(低质量) | 仅 CE | `False` | `seamv1-pseudovoc-A` |
| B | CAM 阈值(低质量) | CE + 0.1 · L_inter | `True` | `seamv1-pseudovoc-B` |

设置 `SEED` 环境变量时会自动在实验名后追加 `-s2` / `-s3`(见 `config.py`)。

## 运行前的两处手动修改

`experiment/seamv1-pseudovoc/` 里保留了原工作站的两处绝对路径:

1. **`train.py`** — `sys.path.insert(0, 'F:/ImageSegmentation/SEAM-REV-V1')`:
   指向分类侧的 `seam-rev/` 目录(提供共享的 `spfr` 模块)。改这一行,
   或者不改代码、运行前设置
   `PYTHONPATH=/path/to/repo/seam-rev:$PYTHONPATH`(失效的 `sys.path`
   条目无害,Python 会静默跳过);
2. **`config.py`** — `DATA_PSEUDO_GT`:伪标签目录的绝对路径,在分类侧
   生成(命令见根目录 README 第 3 节):
   - DRS 精化(高质量,`pseudo_train_drs`):
     `python train_aff.py --session_name resnet38_aff`(可选,或直接用官方
     预训练权重),然后
     `python infer_aff.py --weights resnet38_aff.pth --cam_dir cam_train --infer_list voc12/train_aug.txt --out_rw pseudo_train_drs`
   - CAM 阈值(低质量,`pseudo_train`):
     `python cam_to_pseudo.py --cam_dir cam_train --out_dir pseudo_train --threshold 0.25`

   臂的选择在 `config.py` 里完成:`USE_SPFR`(`False`=仅 CE 的 A 臂,
   `True`=B 臂)以及 `DATA_PSEUDO_GT` / `EXP_NAME` 的取值。

## 运行

```bash
cd segmentation/experiment/seamv1-pseudovoc
SEED=1 EXP_NAME=seamv1-pseudovoc-DRSB python train.py    # 高质量 B 臂,种子 1
SEED=2 python train.py                                    # 默认 DRSA,种子后缀 -s2
```

训练配置(与论文一致):20000 迭代、batch 2、lr 0.001(poly 衰减,
power 0.9)、weight decay 5e-4、随机裁剪 448、尺度抖动 0.5–1.5、翻转 0.5;
B 臂中 SLIC 超像素(`region_size=25`)在增广后的 448×448 裁剪上在线计算,
SPFR 的超像素间损失使用 margin 1.0。种子同时驱动各随机源(`SEED` 变量
读取时做了 `strip()` 处理,以兼容 Windows 的 `set SEED=2` 尾随空格)。

权重保存到 `segmentation/model/<EXP_NAME>/`,TensorBoard 日志到
`segmentation/log/<EXP_NAME>/`。

## 评估

```bash
cd segmentation
python eval_seg.py     # VOC2012 val 上的逐类 IoU
```

协议:多尺度 {0.5, 0.75, 1.0, 1.25, 1.5, 1.75} + 翻转,**无 CRF**
(`TEST_CRF=False`),与论文"no CRF"口径一致。支撑补充材料表 S2/S3 的
逐类结果存档:`seg_perclass.npy`(控制台摘要:`log/logfile.txt`)。
