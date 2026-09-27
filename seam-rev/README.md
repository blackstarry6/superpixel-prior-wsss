# SEAM + SPFR

**Superpixel Prototype Feature Regularization** for Weakly Supervised Semantic Segmentation.

This repository extends [SEAM](https://github.com/YudeWang/SEAM) (CVPR 2020) with SPFR — a plug-and-play module that imposes spatial-semantic structure on backbone features during classification network training.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

### 2. Prepare VOC2012 data
```bash
ln -s /path/to/VOCdevkit/VOC2012 VOC2012
```

### 3. Download pretrained weights
Download WideResNet-38 ImageNet pretrained weights from:
- [Google Drive](https://drive.google.com/open?id=1jWsV5Yev-PwKgvvtUM3GnY0ogb50-qKa)
- [Baidu Cloud](https://pan.baidu.com/s/1ymaMeF0ASjQ9oCGI9cmqHQ) (code: 6nmo)

### 4. Train SEAM + SPFR (legacy full-training recipe)
```bash
python train_SEAM_SPFR.py \
    --voc12_root VOC2012 \
    --weights resnet38_pretrained.pth \
    --session_name seam_spfr_v1 \
    --batch_size 4 \
    --lambda_spfr 0.1 \
    --spfr_warmup 2
```

### 4b. Controlled fine-tuning runs (paper Sec. 4, Experiment 1)
These are the exact commands for the 6 controlled runs reported in the paper
(3 seeds × {FT-only, FT+SPFR}; batch=2, 4 epochs, warmup 1/99, seed protocol):

```bash
# FT-only control (seeds 1, 2, 3)
python train_finetune.py --seed 1 --batch_size 2 --max_epoches 4 --spfr_warmup 99
python train_finetune.py --seed 2 --batch_size 2 --max_epoches 4 --spfr_warmup 99
python train_finetune.py --seed 3 --batch_size 2 --max_epoches 4 --spfr_warmup 99

# FT + SPFR treatment (seeds 1, 2, 3)
python train_finetune.py --seed 1 --batch_size 2 --max_epoches 4 --spfr_warmup 1
python train_finetune.py --seed 2 --batch_size 2 --max_epoches 4 --spfr_warmup 1
python train_finetune.py --seed 3 --batch_size 2 --max_epoches 4 --spfr_warmup 1
```

Evaluation (threshold scan + hash verification):
```bash
python infer_epoch.py --weights <checkpoint> --infer_list voc12/val.txt
python eval_seeds.py
```

### 5. Generate CAM pseudo-labels
```bash
python infer_SEAM.py \
    --weights seam_spfr_v1.pth \
    --infer_list voc12/train_aug.txt \
    --out_cam cam_output/ \
    --out_crf crf_output/
```

### 6. Train segmentation network (DeepLab-v2)
Use the generated pseudo-labels to train a standard segmentation network.

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lambda_spfr` | 0.1 | Overall SPFR weight |
| `--lambda_intra` | 0.1 | Intra-superpixel compactness |
| `--lambda_inter` | 0.05 | Inter-superpixel contrast |
| `--spfr_warmup` | 2 | Epochs before SPFR activates |
| `--sp_n_segments` | 300 | Target superpixel count |
| `--sp_region_size` | 20 | SLIC region size |
| `--sp_margin` | 1.0 | FG-BG push margin |
| `--sp_threshold` | 0.3 | CAM confidence threshold |

## Directory Structure
```
SEAM-REV-V1/
├── train_SEAM_SPFR.py      # ★ SPFR-enhanced training
├── infer_SEAM.py            # CAM inference (unchanged from SEAM)
├── evaluation.py            # mIoU evaluation
├── network/
│   ├── resnet38d.py         # WideResNet-38 backbone
│   ├── resnet38_SEAM.py     # ★ SEAM + SPFR (return_feat)
│   └── resnet38_aff.py      # AffinityNet
├── spfr/                    # ★ SPFR module
│   ├── superpixel.py        # SLIC superpixel generation
│   ├── pooling.py           # GPU-vectorized prototype pooling
│   ├── entropy.py           # CAM entropy for adaptive weighting
│   └── loss.py              # L_intra + entropy-weighted L_inter
├── tool/                    # Utility functions
└── voc12/                   # Data loading & labels
```

## Citation
If you use this code, please cite both SEAM and SPFR:
```
@InProceedings{Wang_2020_CVPR_SEAM,
    author = {Yude Wang and Jie Zhang and Meina Kan and Shiguang Shan and Xilin Chen},
    title = {Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation},
    booktitle = {CVPR},
    year = {2020}
}
```
