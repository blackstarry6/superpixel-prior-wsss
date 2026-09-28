# Segmentation Stage (Experiment 2) — Run Instructions

English | [简体中文](./README_for_SEG_zh-CN.md)

DeepLabV1 (ResNet-38 backbone, initialized from the SEAM classification
weights) trained on pseudo labels produced by the classification stage,
with the optional SPFR loss at the segmentation insertion point.

The framework is inherited from
[semantic-segmentation-codebase](https://github.com/YudeWang/semantic-segmentation-codebase)
(see `README.md` and `LICENSE` in this directory for the upstream project);
this file documents **this study's** experiment matrix and how to run it.

## Experiment matrix (4 arms × 3 seeds)

| Arm | Pseudo labels | Loss | `USE_SPFR` | `EXP_NAME` |
|---|---|---|---|---|
| DRSA | DRS-refined (high quality) | CE only | `False` | `seamv1-pseudovoc-DRSA` (default) |
| DRSB | DRS-refined (high quality) | CE + 0.1 · L_inter | `True` | `seamv1-pseudovoc-DRSB` |
| A | CAM-threshold (low quality) | CE only | `False` | `seamv1-pseudovoc-A` |
| B | CAM-threshold (low quality) | CE + 0.1 · L_inter | `True` | `seamv1-pseudovoc-B` |

Seeds append `-s2` / `-s3` to the experiment name automatically when the
`SEED` environment variable is set (see `config.py`).

## Two manual edits before running

`experiment/seamv1-pseudovoc/` contains two absolute paths from the original
workstation:

1. **`train.py`** — `sys.path.insert(0, 'F:/ImageSegmentation/SEAM-REV-V1')`:
   points to the classification-side `seam-rev/` directory that provides the
   shared `spfr` module. Either edit the line, or leave it and export
   `PYTHONPATH=/path/to/repo/seam-rev:$PYTHONPATH` (a dead `sys.path` entry
   is harmless).
2. **`config.py`** — `DATA_PSEUDO_GT`: absolute path to the pseudo-label
   directory, generated on the classification side (commands in the root
   README, Section 3):
   - DRS-refined (high quality, `pseudo_train_drs`):
     `python train_aff.py --session_name resnet38_aff` (optional, or use the
     official pretrained weights), then
     `python infer_aff.py --weights resnet38_aff.pth --cam_dir cam_train --infer_list voc12/train_aug.txt --out_rw pseudo_train_drs`
   - CAM-threshold (low quality, `pseudo_train`):
     `python cam_to_pseudo.py --cam_dir cam_train --out_dir pseudo_train --threshold 0.25`
   Point `DATA_PSEUDO_GT` to wherever you generated them.

Arm selection is by `config.py`: `USE_SPFR` (`False` = CE-only A arm,
`True` = B arm) and the `DATA_PSEUDO_GT` / `EXP_NAME` values.

## Run

```bash
cd segmentation/experiment/seamv1-pseudovoc
SEED=1 EXP_NAME=seamv1-pseudovoc-DRSB python train.py    # high-quality B arm, seed 1
SEED=2 python train.py                                    # defaults: DRSA, seed suffix -s2
```

Training configuration (as used in the manuscript): 20,000 iterations,
batch 2, lr 0.001 with poly decay (power 0.9), weight decay 5e-4, random
crop 448, scale jitter 0.5–1.5, flip 0.5; in the B arms, SLIC superpixels
(`region_size=25`) are computed online on the augmented 448×448 crops and the
SPFR inter-superpixel loss uses margin 1.0. The seed also drives the RNGs
(the `SEED` variable is read with `strip()` to survive Windows `set SEED=2`).

Checkpoints are written to `segmentation/model/<EXP_NAME>/`, TensorBoard
logs to `segmentation/log/<EXP_NAME>/`.

## Evaluation

```bash
cd segmentation
python eval_seg.py     # per-class IoU on VOC2012 val
```

Protocol: multi-scale {0.5, 0.75, 1.0, 1.25, 1.5, 1.75} + flip, **no CRF**
(`TEST_CRF=False`), matching the manuscript's "no CRF" numbers. Archived
per-class results backing Supplementary Tables S2/S3: `seg_perclass.npy`
(console summary: `log/logfile.txt`).
