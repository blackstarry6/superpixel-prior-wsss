# 弱监督语义分割中的超像素先验 —— 代码与验证记录

[English](./README.md) | 简体中文

本仓库是论文《*"When Do Superpixel Priors Help Weakly Supervised Semantic
Segmentation? A Controlled Study with Feature-Space Diagnostics and a
Projection-Head Intervention"*》的代码与验证记录(论文投稿中)。

研究在受控同种子协议下,检验**超像素原型特征正则(SPFR)**在两个插入点
(SEAM 分类网络训练、DeepLabV1 分割网络训练)能否改进弱监督语义分割,
并通过特征空间探针对零效应进行定位与归因。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `seam-rev/` | 分类阶段:SEAM + SPFR。受控微调运行(`train_finetune.py`)、CAM 推断(`infer_epoch.py`,三尺度 + 翻转)、阈值扫描与锚点校验(`eval_seeds.py`、`verify_anchors.py`)、特征探针(`analyze_probe*.py`、`analyze_topk_probe.py`)、SLIC 纯度测量(`analyze_slic_purity*.py`),以及 SPFR 模块本体(`spfr/`)。 |
| `seam-rev/analysis/` | 论文补充材料各表对应的文本结果存档(对照关系见下表)。 |
| `seam-rev/*_proj.pth` | 投影头小权重(每个约 0.5 MB),支撑补充材料表 S10 的有效秩 / 坍缩检查。 |
| `segmentation/` | 分割阶段:在 DRS 精化伪标签上训练的 DeepLabV1,通过 `SEED` 环境变量控制种子,含可选 SPFR 损失(`experiment/seamv1-pseudovoc/`)。 |

完整模型权重(每个约 420 MB)、CAM 缓存与特征缓存**不纳入**版本管理;
CAM 可由训练入口和 `infer_epoch.py` 重新生成(见下方验证流程)。

## 快速开始

### 1. 分类阶段(实验 1)

```bash
cd seam-rev
pip install -r requirements.txt
ln -s /path/to/VOCdevkit/VOC2012 VOC2012   # 数据集不再分发
```

WideResNet-38 预训练权重:见 `seam-rev/README.md`(官方 SEAM 下载链接)。
论文的六个受控运行(3 种子 × {FT-only, FT+SPFR},按种子配对):

```bash
python train_finetune.py --seed 1 --batch_size 2 --max_epoches 4 --spfr_warmup 99   # FT-only 对照臂
python train_finetune.py --seed 1 --batch_size 2 --max_epoches 4 --spfr_warmup 1    # FT+SPFR 处理臂
# --seed 2、--seed 3 依此类推
```

种子协议:`--seed` 统一固定 Python / NumPy / PyTorch CPU+CUDA 四路随机源,
并设置 `cudnn.deterministic=True`,使同一种子的两臂直接可配对比较。

### 2. 验证流程

```bash
python infer_epoch.py <ckpt> --out cam_<run>   # 重新生成 CAM(0.5/1.0/1.5 + 翻转)
python eval_seeds.py                            # 阈值扫描、逐类 IoU
python verify_anchors.py                        # 复核论文报告的锚点数值
```

`verify_anchors.py` 会对 CAM 缓存重跑评估并断言论文中的锚点 mIoU
(baseline 51.23 @ t=0.25;FT-only 51.16/51.06/51.40,FT+SPFR 51.23/51.07/51.56
@ t=0.15;配对差 +0.07/+0.01/+0.17)。

### 3. 分割阶段(实验 2)

> **⚠️ 前置条件:** `segmentation/experiment/seamv1-pseudovoc/train.py`
> 中有一处绝对路径(搜索 `sys.path.insert`),指向分类侧的 `seam-rev/`
> 目录,用于引入共享的 `spfr` 模块。两种处理方式任选:
>
> 1. 把该行改成你本地的 `seam-rev/` 路径;
> 2. 不改代码,运行前设置环境变量:
>    ```bash
>    export PYTHONPATH=/path/to/repo/seam-rev:$PYTHONPATH
>    ```
>
> (该硬编码路径在不存在的机器上是无害的——Python 会静默跳过失效的
> `sys.path` 条目。)

种子由环境变量选择:

```bash
SEED=1 python train.py   # SEED=2、SEED=3 同理;各臂与实验名见 config.py
```

### 4. 结果 ↔ 论文对照

| 文件 | 对应内容 |
|---|---|
| `seam-rev/analysis/threshold_robustness.txt` | 补充材料表 S4(固定阈值稳健性) |
| `seam-rev/analysis/scan_results_seeds_proj_3seed.txt` | 补充材料表 S5(投影头干预) |
| `seam-rev/analysis/topk_probe.txt` | 补充材料表 S6(top-K 采样探针) |
| `seam-rev/analysis/slic_purity_matrix_*.txt` | 补充材料表 S9 / 图 S1(SLIC 纯度) |
| `seam-rev/scan_results_seeds.txt`、`same_protocol_perclass.npy` | 补充材料表 S1/S7/S8(逐类 IoU) |
| `seam-rev/*_proj.pth` | 补充材料表 S10(有效秩 / 坍缩检查) |
| `seam-rev/finetune_*.log` | 受控运行训练控制台日志 |
| `segmentation/seg_perclass.npy`、`segmentation/log/logfile.txt` | 补充材料表 S2/S3(分割逐类结果) |

## 许可证

本仓库新增部分(SPFR 模块、受控运行与验证工具、分析脚本)采用 MIT;
所扩展的两个上游框架(SEAM、semantic-segmentation-codebase)为其作者的
MIT 许可,声明已保留——见根目录 `LICENSE` 与 `segmentation/LICENSE`。

## 引用

对应论文在审,录用后将补充引用条目。在此之前使用本代码,请引用两个上游
项目(SEAM, CVPR 2020;semantic-segmentation-codebase),并以标题指代本论文。
