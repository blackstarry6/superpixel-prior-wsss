# Superpixel Priors in WSSS — Code and Verification Records

English | [简体中文](./README.zh-CN.md)

Code and verification records for the manuscript *"When Do Superpixel Priors
Help Weakly Supervised Semantic Segmentation? A Controlled Study with
Feature-Space Diagnostics and a Projection-Head Intervention"*.

The study tests, under a controlled same-seed protocol, whether superpixel
prototype feature regularization (SPFR) improves weakly supervised semantic
segmentation at two insertion points (SEAM classification training and
DeepLabV1 segmentation training), and localizes the null result through
feature-space probes.

## Repository structure

| Path | Content |
|---|---|
| `seam-rev/` | Classification stage: SEAM + SPFR. Controlled fine-tuning runs (`train_finetune.py`), CAM inference (`infer_epoch.py`, 3 scales + flip), threshold scans and anchor verification (`eval_seeds.py`, `verify_anchors.py`), feature probes (`analyze_probe*.py`, `analyze_topk_probe.py`), SLIC purity measurement (`analyze_slic_purity*.py`), and the SPFR module (`spfr/`). |
| `seam-rev/analysis/` | Archived textual outputs behind the manuscript's supplementary tables (see mapping below). |
| `seam-rev/*_proj.pth` | Small projection-head checkpoints (≈0.5 MB each) backing the effective-rank / collapse check of Supplementary Table S10. |
| `segmentation/` | Segmentation stage: DeepLabV1 trained on DRS-refined pseudo labels, seed-controlled via the `SEED` environment variable, with optional SPFR loss (`experiment/seamv1-pseudovoc/`). |

Full-model checkpoints (≈420 MB each), CAM caches, and feature caches are not
tracked; CAMs can be regenerated from the training entry points and
`infer_epoch.py` (see the verification flow below).

## Quick start

### 1. Classification stage (Experiment 1)

```bash
cd seam-rev
pip install -r requirements.txt
ln -s /path/to/VOCdevkit/VOC2012 VOC2012   # dataset is not redistributed
```

Pretrained WideResNet-38 weights: see `seam-rev/README.md` (official SEAM
links). The six controlled runs of the manuscript (3 seeds × {FT-only,
FT+SPFR}, paired by seed):

```bash
python train_finetune.py --seed 1 --batch_size 2 --max_epoches 4 --spfr_warmup 99   # FT-only
python train_finetune.py --seed 1 --batch_size 2 --max_epoches 4 --spfr_warmup 1    # FT+SPFR
# repeat with --seed 2, --seed 3
```

Seed protocol: `--seed` fixes Python / NumPy / PyTorch CPU+CUDA RNGs and sets
`cudnn.deterministic=True`, so the two arms of each seed are directly paired.

### 2. Verification flow

```bash
python infer_epoch.py <ckpt> --out cam_<run>   # regenerate CAMs (0.5/1.0/1.5 + flip)
python eval_seeds.py                            # threshold scan, per-class IoU
python verify_anchors.py                        # re-check the archived anchor values
```

`verify_anchors.py` re-runs the evaluation on the CAM caches and asserts the
anchor mIoU values reported in the manuscript (baseline 51.23 @ t=0.25;
FT-only 51.16/51.06/51.40, FT+SPFR 51.23/51.07/51.56 @ t=0.15; paired
deltas +0.07/+0.01/+0.17).

### 3. Segmentation stage (Experiment 2)

> **⚠️ Prerequisite:** `segmentation/experiment/seamv1-pseudovoc/train.py`
> contains one absolute path (search for `sys.path.insert`) pointing to the
> classification-side `seam-rev/` directory, which provides the shared `spfr`
> module. Either edit that line to your local path, or leave it untouched and
> export it instead:
>
> ```bash
> export PYTHONPATH=/path/to/repo/seam-rev:$PYTHONPATH
> ```
>
> (The hardcoded entry is harmless on machines where the path does not exist —
> Python silently skips missing `sys.path` entries.)

Seeds are selected by environment variable:

```bash
SEED=1 python train.py   # also SEED=2, SEED=3; arms and experiment names in config.py
```

### 4. Results ↔ manuscript mapping

| File | Backs |
|---|---|
| `seam-rev/analysis/threshold_robustness.txt` | Suppl. Table S4 (fixed-threshold robustness) |
| `seam-rev/analysis/scan_results_seeds_proj_3seed.txt` | Suppl. Table S5 (projection-head intervention) |
| `seam-rev/analysis/topk_probe.txt` | Suppl. Table S6 (top-K sampling probe) |
| `seam-rev/analysis/slic_purity_matrix_*.txt` | Suppl. Table S9 / Fig. S1 (SLIC purity) |
| `seam-rev/scan_results_seeds.txt`, `same_protocol_perclass.npy` | Suppl. Tables S1/S7/S8 (per-class IoU) |
| `seam-rev/*_proj.pth` | Suppl. Table S10 (effective rank / collapse) |
| `seam-rev/finetune_*.log` | Training console logs of the controlled runs |
| `segmentation/seg_perclass.npy`, `segmentation/log/logfile.txt` | Suppl. Tables S2/S3 (segmentation per-class) |

## License

MIT for the additions in this repository (SPFR module, controlled-run and
verification tooling, analysis scripts); the two upstream frameworks (SEAM,
semantic-segmentation-codebase) are MIT-licensed by their authors and their
notices are retained — see `LICENSE` and `segmentation/LICENSE`.

## Citation

The accompanying manuscript is under review; a citation entry will be added
upon acceptance. If you use this code before then, please cite both upstream
projects (SEAM, CVPR 2020; semantic-segmentation-codebase) and refer to the
manuscript by title.
