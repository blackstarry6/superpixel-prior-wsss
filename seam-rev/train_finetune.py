"""
SEAM + SPFR Fine-tuning Script (Final Params)
==============================================
Fine-tunes SEAM pretrained model with SPFR for 4 epochs.

Key differences from train_SEAM_SPFR.py:
  - Loads SEAM pretrained weights (resnet38_SEAM.pth) as initialization
  - Freezes conv1a, b2, b3 (shallow layers)
  - Reduced lr_multipliers: fc8 from 20x → 5x
  - Conservative SPFR: λ_intra=0.02, λ_inter=0.05, threshold=0.20
  - Saves checkpoint after each epoch for per-epoch CAM evaluation

Usage:
    python train_finetune.py --session_name finetune_v1
"""

import numpy as np
import torch
import random
import cv2
import os
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils, visualization
import argparse
import importlib
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from spfr import spfr_loss, generate_superpixels, resize_superpixels


# ── SEAM helper functions ────────────────────────────────────────

def adaptive_min_pooling_loss(x):
    n, c, h, w = x.size()
    k = h * w // 4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n, -1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y) / (k * n)
    return loss


def max_onehot(x):
    n, c, h, w = x.size()
    x_max = torch.max(x[:, 1:, :, :], dim=1, keepdim=True)[0]
    x[:, 1:, :, :][x[:, 1:, :, :] != x_max] = 0
    return x


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SEAM + SPFR Fine-tuning')

    parser.add_argument("--batch_size", default=2, type=int)  # paper protocol: batch=2
    parser.add_argument("--max_epoches", default=4, type=int)
    parser.add_argument("--network", default="network.resnet38_SEAM", type=str)
    parser.add_argument("--lr", default=0.005, type=float)
    parser.add_argument("--num_workers", default=0, type=int,
                        help="Windows: must be 0 (multiprocessing spawn issue)")
    parser.add_argument("--wt_dec", default=1e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--session_name", default="finetune_spfr", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--weights", default="../SEAM_model/resnet38_SEAM.pth", type=str)
    parser.add_argument("--voc12_root", default='VOC2012', type=str)
    parser.add_argument("--tblog_dir", default='./tblog_finetune', type=str)
    parser.add_argument("--seed", default=1, type=int,
                        help="random seed; 同一 seed 下 FT-only 与 FT+SPFR 成对运行，保证 Δ 可归因")

    # SPFR params (final)
    parser.add_argument("--lambda_spfr", default=0.1, type=float)
    parser.add_argument("--lambda_intra", default=0.02, type=float)
    parser.add_argument("--lambda_inter", default=0.05, type=float)
    parser.add_argument("--spfr_warmup", default=1, type=int)
    parser.add_argument("--sp_n_segments", default=300, type=int)
    parser.add_argument("--sp_region_size", default=20, type=int)
    parser.add_argument("--sp_margin", default=1.0, type=float)
    parser.add_argument("--sp_threshold", default=0.20, type=float)
    parser.add_argument("--sp_min_pixels", default=10, type=int)

    # ── H2 intervention: projection head (added 2026-09-05) ──────
    # use_proj_head=1: SPFR loss operates on 128-d projected features
    #   (1×1 conv + ReLU, PCC-style) instead of raw conv5 (1024-d).
    # use_proj_head=0 (default): original conv5 path (all existing runs).
    # To revert: set default back to 0 and uncomment the original
    # spfr_loss call below (both paths kept).
    parser.add_argument("--use_proj_head", default=0, type=int,
                        help="1=SPFR on 128-d projection head (H2 intervention); 0=original conv5")

    args = parser.parse_args()

    # ── Reproducibility: seed all RNG sources ─────────────────────
    # num_workers=0 → shuffle 与数据增强都在主进程消耗 RNG，四路种子 +
    # cudnn.deterministic 即可覆盖全部随机源（shuffle/增强/CUDA 算子）。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'Random seed: {args.seed}')

    # ── Logging ──────────────────────────────────────────────────
    pyutils.Logger(args.session_name + '.log')
    print('=' * 60)
    print('  SEAM + SPFR Fine-tuning (Final Params)')
    print('=' * 60)
    print(vars(args))

    # ── Model ────────────────────────────────────────────────────
    model = getattr(importlib.import_module(args.network), 'Net')()
    print(f'\nModel: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params')

    tblogger = SummaryWriter(args.tblog_dir)

    # ── Load SEAM pretrained weights ─────────────────────────────
    print(f'\nLoading SEAM pretrained weights: {args.weights}')
    state_dict = torch.load(args.weights, map_location='cpu')
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k.replace('module.', '')] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f'  Missing keys: {len(missing)}')
    if unexpected:
        print(f'  Unexpected keys: {len(unexpected)}')
    print(f'  Weights loaded OK (strict=False)')

    # ── H2 intervention: projection head (PCC-style 1×1 conv + ReLU) ──
    # Maps conv5 (1024-d) → 128-d projected space. SPFR loss is computed
    # on this projected space when --use_proj_head=1, isolating the
    # "operating space" variable (single-variable intervention).
    proj_head = None
    if args.use_proj_head:
        import torch.nn as nn
        proj_head = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1),
            nn.ReLU(inplace=True),
        ).cuda()
        proj_head.train()
        # Add to optimizer so it learns during fine-tuning
        print(f'\n  [H2] Projection head enabled: conv5(1024) → proj(128), '
              f'{sum(p.numel() for p in proj_head.parameters())} params')
    else:
        print(f'\n  [H2] Projection head disabled (original conv5 path)')

    # ── Freeze shallow layers ────────────────────────────────────
    freeze_names = ['conv1a', 'b2', 'b2_1', 'b2_2']
    frozen_params = 0
    for name, param in model.named_parameters():
        for fn in freeze_names:
            if name.startswith(fn):
                param.requires_grad = False
                frozen_params += param.numel()
                break
    print(f'  Frozen: {frozen_params/1e6:.1f}M params ({freeze_names})')
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Trainable: {trainable/1e6:.1f}M params')

    # ── Data ─────────────────────────────────────────────────────
    train_dataset = voc12.data.VOC12ClsDataset(
        args.train_list, voc12_root=args.voc12_root,
        transform=transforms.Compose([
            imutils.RandomResizeLong(448, 768),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),
            np.asarray,
            model.normalize,
            imutils.RandomCrop(args.crop_size),
            imutils.HWC_to_CHW,
            torch.from_numpy,
        ]))

    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    max_step = len(train_dataset) // args.batch_size * args.max_epoches
    print(f'\nTraining: {len(train_dataset)} images, {args.batch_size} batch, '
          f'{args.max_epoches} epochs, {max_step} steps')

    # ── Optimizer with adjusted lr multipliers ───────────────────
    # Manual param groups: [backbone(b4+), b5+b6, fc8+PCM, biases]
    # multipliers: [1x, 2x, 5x, 5x]
    backbone_params = []
    mid_params = []
    head_params = []
    bias_params = []

    # "from_scratch" 指这些层在 SEAM 预训练阶段从零训练（不在 ImageNet 权重中），
    # 而非 fine-tune 阶段重新初始化——它们从 resnet38_SEAM.pth 完整加载（missing=0）。
    # 给 5x 学习率是因为它们是 SEAM 训练中改动最大的层。
    head_layer_names = ['fc8', 'f8_3', 'f8_4', 'f9']
    mid_layer_names = ['b5', 'b6', 'b7', 'bn7']       # 2x

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(ln) for ln in head_layer_names):
            if 'bias' in name:
                bias_params.append(param)
            else:
                head_params.append(param)
        elif any(name.startswith(ln) for ln in mid_layer_names):
            if 'bias' in name:
                bias_params.append(param)
            else:
                mid_params.append(param)
        else:
            if 'bias' in name:
                bias_params.append(param)
            else:
                backbone_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': mid_params, 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': head_params, 'lr': 5 * args.lr, 'weight_decay': args.wt_dec},
        {'params': bias_params, 'lr': 5 * args.lr, 'weight_decay': 0},
    ]
    # ── H2: add projection head params to optimizer (same lr as mid_params) ──
    if args.use_proj_head and proj_head is not None:
        proj_params = list(proj_head.parameters())
        param_groups.append({'params': proj_params, 'lr': 2 * args.lr, 'weight_decay': 0})
        print(f'  [H2] Added projection head to optimizer: '
              f'{sum(p.numel() for p in proj_params)} params, lr={2*args.lr}')
    for i, pg in enumerate(param_groups):
        n = sum(p.numel() for p in pg['params'])
        print(f'  Group {i}: lr={pg["lr"]:.4f}, params={n/1e6:.1f}M')

    optimizer = torchutils.PolyOptimizer(
        param_groups, lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    # ── Meters ───────────────────────────────────────────────────
    avg_meter = pyutils.AverageMeter(
        'loss', 'loss_cls', 'loss_er', 'loss_ecr', 'loss_spfr'
    )

    timer = pyutils.Timer("Session started: ")
    best_spfr_loss = float('inf')

    # ═══════════════════════════════════════════════════════════════
    #  Training Loop
    # ═══════════════════════════════════════════════════════════════
    for ep in range(args.max_epoches):

        spfr_active = (ep >= args.spfr_warmup)
        epoch_spfr_sum = 0.0
        epoch_spfr_count = 0
        epoch_intra_sum = 0.0
        epoch_inter_sum = 0.0

        for iter, pack in enumerate(train_data_loader):

            scale_factor = 0.3
            img1 = pack[1]
            img2 = F.interpolate(img1, scale_factor=scale_factor,
                                 mode='bilinear', align_corners=True)
            N, C, H, W = img1.size()
            label = pack[2]
            bg_score = torch.ones((N, 1))
            label_padded = torch.cat((bg_score, label), dim=1)
            label_padded = label_padded.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)

            # ── Forward ──────────────────────────────────────────
            cam1, cam_rv1, feat1 = model(img1, return_feat=True)
            cam2, cam_rv2 = model(img2)

            # ── SEAM losses ──────────────────────────────────────
            label1 = F.adaptive_avg_pool2d(cam1, (1, 1))
            loss_rvmin1 = adaptive_min_pooling_loss((cam_rv1 * label_padded)[:, 1:, :, :])
            cam1_scaled = F.interpolate(
                visualization.max_norm(cam1), scale_factor=scale_factor,
                mode='bilinear', align_corners=True) * label_padded
            cam_rv1_scaled = F.interpolate(
                visualization.max_norm(cam_rv1), scale_factor=scale_factor,
                mode='bilinear', align_corners=True) * label_padded

            label2 = F.adaptive_avg_pool2d(cam2, (1, 1))
            loss_rvmin2 = adaptive_min_pooling_loss((cam_rv2 * label_padded)[:, 1:, :, :])
            cam2_norm = visualization.max_norm(cam2) * label_padded
            cam_rv2_norm = visualization.max_norm(cam_rv2) * label_padded

            loss_cls1 = F.multilabel_soft_margin_loss(label1[:, 1:, :, :],
                                                       label_padded[:, 1:, :, :])
            loss_cls2 = F.multilabel_soft_margin_loss(label2[:, 1:, :, :],
                                                       label_padded[:, 1:, :, :])

            ns, cs, hs, ws = cam2_norm.size()
            loss_er = torch.mean(torch.abs(cam1_scaled[:, 1:, :, :] -
                                           cam2_norm[:, 1:, :, :]))

            cam1_bg = cam1_scaled.clone()
            cam1_bg[:, 0, :, :] = 1 - torch.max(cam1_bg[:, 1:, :, :], dim=1)[0]
            cam2_bg = cam2_norm.clone()
            cam2_bg[:, 0, :, :] = 1 - torch.max(cam2_bg[:, 1:, :, :], dim=1)[0]

            tensor_ecr1 = torch.abs(max_onehot(cam2_bg.detach()) - cam_rv1_scaled)
            tensor_ecr2 = torch.abs(max_onehot(cam1_bg.detach()) - cam_rv2_norm)
            loss_ecr1 = torch.mean(torch.topk(
                tensor_ecr1.view(ns, -1), k=int(21 * hs * ws * 0.2), dim=-1)[0])
            loss_ecr2 = torch.mean(torch.topk(
                tensor_ecr2.view(ns, -1), k=int(21 * hs * ws * 0.2), dim=-1)[0])
            loss_ecr = loss_ecr1 + loss_ecr2

            loss_cls = (loss_cls1 + loss_cls2) / 2 + (loss_rvmin1 + loss_rvmin2) / 2
            loss_seam = loss_cls + loss_er + loss_ecr

            # ── SPFR ─────────────────────────────────────────────
            loss_spfr_val = torch.tensor(0.0, device=img1.device)
            l_intra_log = torch.tensor(0.0, device=img1.device)
            l_inter_log = torch.tensor(0.0, device=img1.device)
            n_pairs = 0

            if spfr_active:
                spfr_batch = torch.tensor(0.0, device=img1.device)
                valid_cnt = 0
                for b_idx in range(N):
                    if label[b_idx].sum() < 0.5:
                        continue
                    segments, num_sp = generate_superpixels(
                        img1[b_idx].cpu(), n_segments=args.sp_n_segments,
                        region_size=args.sp_region_size)
                    feat_h, feat_w = feat1.shape[2], feat1.shape[3]
                    segments_ds = resize_superpixels(segments, feat_h, feat_w).to(img1.device)

                    # ── H2 intervention: choose operating space ─────
                    # Original path (all existing runs): SPFR on raw conv5
                    # feat_spfr = feat1[b_idx:b_idx+1]
                    #
                    # New path (--use_proj_head=1): SPFR on 128-d projection
                    if args.use_proj_head and proj_head is not None:
                        feat_spfr = proj_head(feat1[b_idx:b_idx+1])
                    else:
                        feat_spfr = feat1[b_idx:b_idx+1]  # original conv5 (1024-d)

                    l_spfr, l_intra, l_inter, pairs = spfr_loss(
                        feat_spfr, cam1[b_idx:b_idx+1],
                        segments_ds.unsqueeze(0), label[b_idx:b_idx+1],
                        lambda_intra=args.lambda_intra,
                        lambda_inter=args.lambda_inter,
                        margin=args.sp_margin,
                        threshold=args.sp_threshold,
                        min_pixels=args.sp_min_pixels,
                    )
                    # Force GPU — spfr_loss may return CPU tensors in edge cases
                    l_spfr  = torch.as_tensor(l_spfr,  device=img1.device)
                    l_intra = torch.as_tensor(l_intra, device=img1.device)
                    l_inter = torch.as_tensor(l_inter, device=img1.device)

                    spfr_batch += l_spfr
                    l_intra_log += l_intra
                    l_inter_log += l_inter
                    n_pairs += pairs
                    valid_cnt += 1

                if valid_cnt > 0:
                    loss_spfr_val = spfr_batch / valid_cnt
                    epoch_spfr_sum += loss_spfr_val.item()
                    epoch_spfr_count += 1
                    epoch_intra_sum += (l_intra_log / valid_cnt).item()
                    epoch_inter_sum += (l_inter_log / valid_cnt).item()

            # ── Total ────────────────────────────────────────────
            if spfr_active and n_pairs > 0:
                loss = loss_seam + args.lambda_spfr * loss_spfr_val
            else:
                loss = loss_seam

            # ── Backward ─────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ── Logging ──────────────────────────────────────────
            log_dict = {
                'loss': loss.item(),
                'loss_cls': loss_cls.item(),
                'loss_er': loss_er.item(),
                'loss_ecr': loss_ecr.item(),
                'loss_spfr': loss_spfr_val.item() if n_pairs > 0 else 0.0,
            }
            avg_meter.add(log_dict)

            if (optimizer.global_step - 1) % 100 == 0:
                timer.update_progress(optimizer.global_step / max_step)
                spfr_str = ''
                if spfr_active and n_pairs > 0:
                    spfr_str = (f' SPFR={loss_spfr_val.item():.4f}'
                                f'(intra={l_intra_log.item()/max(valid_cnt,1):.4f}'
                                f' inter={l_inter_log.item()/max(valid_cnt,1):.4f})'
                                f' ratio={loss_spfr_val.item()/max(loss_cls.item(),1e-8):.3f}')
                print(f'Ep{ep+1} Iter{optimizer.global_step-1:5d}/{max_step} '
                      f'L={avg_meter.get("loss"):.4f} '
                      f'cls={loss_cls.item():.3f} '
                      f'er={loss_er.item():.4f} '
                      f'ecr={loss_ecr.item():.4f}'
                      f'{spfr_str} '
                      f'({(iter+1)*args.batch_size/timer.get_stage_elapsed():.1f} im/s)',
                      flush=True)
                avg_meter.pop()

        # ── End of epoch ─────────────────────────────────────────
        print(f'\n=== Epoch {ep+1} Complete ===')
        if epoch_spfr_count > 0:
            avg_intra = epoch_intra_sum / epoch_spfr_count
            avg_inter = epoch_inter_sum / epoch_spfr_count
            avg_spfr   = epoch_spfr_sum / epoch_spfr_count
            print(f'  SPFR avg: {avg_spfr:.4f} (intra={avg_intra:.4f}, inter={avg_inter:.4f})')

        # Save checkpoint
        ckpt_path = f'{args.session_name}_ep{ep+1}.pth'
        torch.save(model.module.state_dict(), ckpt_path)
        print(f'  Checkpoint saved: {ckpt_path}')
        # ── H2: also save projection head weights (for post-hoc space analysis) ──
        if args.use_proj_head and proj_head is not None:
            proj_path = f'{args.session_name}_ep{ep+1}_proj.pth'
            torch.save(proj_head.state_dict(), proj_path)
            print(f'  [H2] Projection head saved: {proj_path}')

    # ── Final save ───────────────────────────────────────────────
    final_path = f'{args.session_name}.pth'
    torch.save(model.module.state_dict(), final_path)
    print(f'\nFinal model: {final_path}')
    # ── H2: final projection head save ──
    if args.use_proj_head and proj_head is not None:
        proj_final = f'{args.session_name}_proj.pth'
        torch.save(proj_head.state_dict(), proj_final)
        print(f'  [H2] Final projection head: {proj_final}')
    print('Training complete.')
