"""
SEAM + SPFR Training Script
============================
Extension of SEAM (CVPR 2020) with Superpixel Prototype Feature Regularization.

Key changes from original train_SEAM.py:
  1. Model forward returns conv5 features (return_feat=True)
  2. Online SLIC superpixel generation per image
  3. SPFR Loss added to total loss with warmup schedule

Total Loss = L_cls + L_er + L_ecr + λ_spfr * L_spfr

Usage:
    python train_SEAM_SPFR.py --voc12_root VOC2012 --weights res38.pth
                             --session_name seam_spfr_v1
                             --lambda_spfr 0.1 --spfr_warmup 2
"""

import numpy as np
import torch
import random
import cv2
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils, visualization
import argparse
import importlib
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from spfr import spfr_loss, generate_superpixels, resize_superpixels


# ── SEAM original helper functions (unchanged) ──────────────────

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


# ── Tensor → numpy image (for superpixel generation) ───────────

def tensor_to_rgb_numpy(img_tensor):
    """
    Convert ImageNet-normalized tensor to uint8 RGB numpy (H, W, 3).

    Args:
        img_tensor: (3, H, W) torch.Tensor, ImageNet-normalized

    Returns:
        img_np: (H, W, 3) uint8 numpy array
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_np = img_tensor.cpu().numpy()
    img_np = img_np * std[:, None, None] + mean[:, None, None]
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    return img_np.transpose(1, 2, 0)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SEAM + SPFR Training')

    # ── SEAM original args ────────────────────────────────────
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=8, type=int)
    parser.add_argument("--network", default="network.resnet38_SEAM", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--session_name", default="resnet38_SEAM_SPFR", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--voc12_root", default='VOC2012', type=str)
    parser.add_argument("--tblog_dir", default='./tblog', type=str)

    # ── SPFR-specific args ────────────────────────────────────
    parser.add_argument("--lambda_spfr", default=0.1, type=float,
                        help="Overall SPFR loss weight")
    parser.add_argument("--lambda_intra", default=0.1, type=float,
                        help="Intra-superpixel compactness weight (relative)")
    parser.add_argument("--lambda_inter", default=0.05, type=float,
                        help="Inter-superpixel contrast weight (relative)")
    parser.add_argument("--spfr_warmup", default=2, type=int,
                        help="Epochs to delay SPFR (CAM needs to stabilize first)")
    parser.add_argument("--sp_n_segments", default=300, type=int,
                        help="Target number of superpixels")
    parser.add_argument("--sp_region_size", default=20, type=int,
                        help="SLIC region_size parameter")
    parser.add_argument("--sp_margin", default=1.0, type=float,
                        help="Margin for FG-BG push in L_inter")
    parser.add_argument("--sp_threshold", default=0.3, type=float,
                        help="CAM confidence threshold for FG/BG assignment")
    parser.add_argument("--sp_min_pixels", default=10, type=int,
                        help="Minimum pixels per superpixel (smaller = ignored)")

    args = parser.parse_args()

    # ── Logging ────────────────────────────────────────────────
    pyutils.Logger(args.session_name + '.log')
    print(vars(args))

    # ── Model ──────────────────────────────────────────────────
    model = getattr(importlib.import_module(args.network), 'Net')()
    print(model)

    tblogger = SummaryWriter(args.tblog_dir)

    # ── Data ───────────────────────────────────────────────────
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

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    # ── Optimizer (SEAM original PolyOptimizer) ────────────────
    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0},
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    # ── Load pretrained weights ────────────────────────────────
    if args.weights[-7:] == '.params':
        import network.resnet38d
        assert 'resnet38' in args.network
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    # ── Meters ─────────────────────────────────────────────────
    avg_meter = pyutils.AverageMeter(
        'loss', 'loss_cls', 'loss_er', 'loss_ecr', 'loss_spfr'
    )

    timer = pyutils.Timer("Session started: ")

    # ═══════════════════════════════════════════════════════════
    #  Training Loop
    # ═══════════════════════════════════════════════════════════
    for ep in range(args.max_epoches):

        spfr_active = (ep >= args.spfr_warmup)

        for iter, pack in enumerate(train_data_loader):

            scale_factor = 0.3
            img1 = pack[1]                                                    # (N, 3, 448, 448)
            img2 = F.interpolate(img1, scale_factor=scale_factor,
                                 mode='bilinear', align_corners=True)         # (N, 3, 134, 134)
            N, C, H, W = img1.size()
            label = pack[2]                                                   # (N, 20)
            bg_score = torch.ones((N, 1))
            label_padded = torch.cat((bg_score, label), dim=1)
            label_padded = label_padded.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)

            # ── Forward pass (with features for SPFR on img1) ──
            cam1, cam_rv1, feat1 = model(img1, return_feat=True)
            cam2, cam_rv2 = model(img2)

            # ── SEAM: Loss components ─────────────────────────
            label1 = F.adaptive_avg_pool2d(cam1, (1, 1))
            loss_rvmin1 = adaptive_min_pooling_loss((cam_rv1 * label_padded)[:, 1:, :, :])
            cam1_scaled = F.interpolate(
                visualization.max_norm(cam1), scale_factor=scale_factor,
                mode='bilinear', align_corners=True
            ) * label_padded
            cam_rv1_scaled = F.interpolate(
                visualization.max_norm(cam_rv1), scale_factor=scale_factor,
                mode='bilinear', align_corners=True
            ) * label_padded

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
            loss_ecr1 = torch.mean(
                torch.topk(tensor_ecr1.view(ns, -1),
                           k=int(21 * hs * ws * 0.2), dim=-1)[0])
            loss_ecr2 = torch.mean(
                torch.topk(tensor_ecr2.view(ns, -1),
                           k=int(21 * hs * ws * 0.2), dim=-1)[0])
            loss_ecr = loss_ecr1 + loss_ecr2

            loss_cls = (loss_cls1 + loss_cls2) / 2 + (loss_rvmin1 + loss_rvmin2) / 2
            loss_seam = loss_cls + loss_er + loss_ecr

            # ── SPFR: Loss ────────────────────────────────────
            loss_spfr_val = torch.tensor(0.0, device=img1.device)
            loss_intra_val = torch.tensor(0.0, device=img1.device)
            loss_inter_val = torch.tensor(0.0, device=img1.device)
            n_pairs = 0

            if spfr_active:
                loss_spfr_batch = 0.0
                intra_sum = 0.0
                inter_sum = 0.0
                pair_sum = 0
                valid_count = 0

                for b_idx in range(N):
                    # Skip images with no foreground classes
                    if label[b_idx].sum() < 0.5:
                        continue

                    # Generate superpixels from original image
                    img_b = img1[b_idx].cpu()  # (3, H, W) on CPU for SLIC
                    segments, num_sp = generate_superpixels(
                        img_b,
                        n_segments=args.sp_n_segments,
                        region_size=args.sp_region_size,
                    )

                    # Resize superpixels to conv5 feature map size
                    feat_h, feat_w = feat1.shape[2], feat1.shape[3]
                    segments_ds = resize_superpixels(segments, feat_h, feat_w)
                    segments_ds = segments_ds.to(img1.device)

                    # Compute SPFR loss for this image
                    l_spfr, l_intra, l_inter, pairs = spfr_loss(
                        feat1[b_idx].unsqueeze(0),      # (1, C, h, w)
                        cam1[b_idx].unsqueeze(0),        # (1, 21, H, W)
                        segments_ds.unsqueeze(0),         # (1, h, w)
                        label[b_idx].unsqueeze(0),        # (1, 20)
                        lambda_intra=args.lambda_intra,
                        lambda_inter=args.lambda_inter,
                        margin=args.sp_margin,
                        threshold=args.sp_threshold,
                        min_pixels=args.sp_min_pixels,
                    )

                    loss_spfr_batch += l_spfr
                    intra_sum += l_intra
                    inter_sum += l_inter
                    pair_sum += pairs
                    valid_count += 1

                if valid_count > 0:
                    loss_spfr_val = loss_spfr_batch / valid_count
                    loss_intra_val = intra_sum / valid_count
                    loss_inter_val = inter_sum / valid_count
                    n_pairs = pair_sum

            # ── Total Loss ─────────────────────────────────────
            if spfr_active and isinstance(loss_spfr_val, torch.Tensor) and loss_spfr_val.requires_grad:
                loss = loss_seam + args.lambda_spfr * loss_spfr_val
            else:
                loss = loss_seam

            # ── Backward ───────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ── Logging ────────────────────────────────────────
            log_dict = {
                'loss': loss.item(),
                'loss_cls': loss_cls.item(),
                'loss_er': loss_er.item(),
                'loss_ecr': loss_ecr.item(),
                'loss_spfr': loss_spfr_val.item() if isinstance(loss_spfr_val, torch.Tensor) else 0.0,
            }
            avg_meter.add(log_dict)

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                spfr_str = ''
                if spfr_active and n_pairs > 0:
                    spfr_str = (f' | SPFR:{loss_spfr_val.item():.4f} '
                                f'(intra:{loss_intra_val.item():.4f} '
                                f'inter:{loss_inter_val.item():.4f} '
                                f'pairs:{n_pairs})')

                print(f'Iter:{optimizer.global_step - 1:5d}/{max_step:5d} '
                      f'loss:{avg_meter.get("loss"):.4f} '
                      f'cls:{avg_meter.get("loss_cls"):.4f} '
                      f'er:{avg_meter.get("loss_er"):.4f} '
                      f'ecr:{avg_meter.get("loss_ecr"):.4f}'
                      f'{spfr_str} '
                      f'imps:{(iter + 1) * args.batch_size / timer.get_stage_elapsed():.1f} '
                      f'Fin:{timer.str_est_finish()} '
                      f'lr:{optimizer.param_groups[0]["lr"]:.4f}',
                      flush=True)

                avg_meter.pop()

                # ── TensorBoard visualization ──────────────────
                img_8 = img1[0].numpy().transpose((1, 2, 0))
                img_8 = np.ascontiguousarray(img_8)
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
                img_8[:, :, 0] = (img_8[:, :, 0] * std[0] + mean[0]) * 255
                img_8[:, :, 1] = (img_8[:, :, 1] * std[1] + mean[1]) * 255
                img_8[:, :, 2] = (img_8[:, :, 2] * std[2] + mean[2]) * 255
                img_8[img_8 > 255] = 255
                img_8[img_8 < 0] = 0
                img_8 = img_8.astype(np.uint8)

                input_img = img_8.transpose((2, 0, 1))
                h = H // 4
                w = W // 4
                p1 = F.interpolate(cam1_scaled, (h, w), mode='bilinear')[0].detach().cpu().numpy()
                p2 = F.interpolate(cam2_norm, (h, w), mode='bilinear')[0].detach().cpu().numpy()
                p_rv1 = F.interpolate(cam_rv1_scaled, (h, w), mode='bilinear')[0].detach().cpu().numpy()
                p_rv2 = F.interpolate(cam_rv2_norm, (h, w), mode='bilinear')[0].detach().cpu().numpy()

                image = cv2.resize(img_8, (w, h), interpolation=cv2.INTER_CUBIC).transpose((2, 0, 1))
                CLS1, CAM1, _, _ = visualization.generate_vis(
                    p1, None, image, func_label2color=visualization.VOClabel2colormap,
                    threshold=None, norm=False)
                CLS2, CAM2, _, _ = visualization.generate_vis(
                    p2, None, image, func_label2color=visualization.VOClabel2colormap,
                    threshold=None, norm=False)
                CLS_RV1, CAM_RV1, _, _ = visualization.generate_vis(
                    p_rv1, None, image, func_label2color=visualization.VOClabel2colormap,
                    threshold=None, norm=False)
                CLS_RV2, CAM_RV2, _, _ = visualization.generate_vis(
                    p_rv2, None, image, func_label2color=visualization.VOClabel2colormap,
                    threshold=None, norm=False)

                loss_dict = {
                    'loss': loss.item(),
                    'loss_cls': loss_cls.item(),
                    'loss_er': loss_er.item(),
                    'loss_ecr': loss_ecr.item(),
                }
                if spfr_active and n_pairs > 0:
                    loss_dict['loss_spfr'] = loss_spfr_val.item()

                itr = optimizer.global_step - 1
                tblogger.add_scalars('loss', loss_dict, itr)
                tblogger.add_scalar('lr', optimizer.param_groups[0]['lr'], itr)
                tblogger.add_image('Image', input_img, itr)
                tblogger.add_image('CLS1', CLS1, itr)
                tblogger.add_image('CLS2', CLS2, itr)
                tblogger.add_image('CLS_RV1', CLS_RV1, itr)
                tblogger.add_image('CLS_RV2', CLS_RV2, itr)
                tblogger.add_images('CAM1', CAM1, itr)
                tblogger.add_images('CAM2', CAM2, itr)
                tblogger.add_images('CAM_RV1', CAM_RV1, itr)
                tblogger.add_images('CAM_RV2', CAM_RV2, itr)

        else:
            print('')
            timer.reset_stage()

    # ── Save model ─────────────────────────────────────────────
    torch.save(model.module.state_dict(), args.session_name + '.pth')
    print(f'\n✅ Training complete. Model saved to {args.session_name}.pth')
