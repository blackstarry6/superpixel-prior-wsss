"""
Superpixel generation for SPFR.

Uses OpenCV SLIC to generate superpixel labels from RGB images.
Operates on numpy arrays (CPU) — called per-image in the training loop.
"""

import numpy as np
import cv2
import torch


def generate_superpixels(image_tensor, n_segments=300, region_size=20,
                         ruler=10.0, n_iter=5, fast=True):
    """
    Generate SLIC superpixel labels from a normalized image tensor.

    Args:
        image_tensor: (3, H, W) torch.Tensor, ImageNet-normalized
        n_segments:   target number of superpixels (approximate)
        region_size:  SLIC region_size parameter
        ruler:        SLIC ruler (compactness) parameter
        n_iter:       SLIC iteration count (reduced for speed in fast mode)
        fast:         If True, downsample to 256px before SLIC for speed (~3x faster)

    Returns:
        segments: (H, W) torch.LongTensor, contiguous superpixel IDs (0..K-1)
        num_sp:    int, number of superpixels
    """
    # Undo ImageNet normalization → uint8 RGB
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_np = image_tensor.cpu().numpy()                     # (3, H, W)
    img_np = img_np * std[:, None, None] + mean[:, None, None]  # unnormalize
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    img_np = img_np.transpose(1, 2, 0)                      # (H, W, 3)

    H_orig, W_orig = img_np.shape[:2]

    # Fast mode: downsample for SLIC, then upsample labels
    if fast and max(H_orig, W_orig) > 256:
        scale = 256.0 / max(H_orig, W_orig)
        H_small = int(round(H_orig * scale))
        W_small = int(round(W_orig * scale))
        img_small = cv2.resize(img_np, (W_small, H_small), interpolation=cv2.INTER_LINEAR)
        # Adjust region_size proportionally
        region_size_small = max(5, int(round(region_size * scale)))
    else:
        img_small = img_np
        H_small, W_small = H_orig, W_orig
        region_size_small = region_size

    # SLIC superpixel segmentation
    slic = cv2.ximgproc.createSuperpixelSLIC(
        img_small,
        algorithm=cv2.ximgproc.SLIC,
        region_size=region_size_small,
        ruler=ruler,
    )
    slic.iterate(n_iter)
    slic.enforceLabelConnectivity()
    raw_labels = slic.getLabels()  # (H_small, W_small) int32

    # Upsample labels back to original resolution (if downsampled)
    if fast and (H_small != H_orig or W_small != W_orig):
        raw_labels = cv2.resize(
            raw_labels.astype(np.float32),
            (W_orig, H_orig),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)

    # Remap to contiguous IDs 0..K-1
    unique = np.unique(raw_labels)
    H, W = raw_labels.shape
    remap = np.zeros(raw_labels.max() + 1, dtype=np.int64)
    for new, old in enumerate(unique):
        remap[old] = new
    labels_remapped = remap[raw_labels]

    segments = torch.from_numpy(labels_remapped).long()
    num_sp = len(unique)

    return segments, num_sp


def resize_superpixels(segments, target_h, target_w):
    """
    Resize superpixel labels to match feature map spatial size.

    Args:
        segments:  (H, W) torch.LongTensor, superpixel IDs
        target_h:  int, target height
        target_w:  int, target width

    Returns:
        (target_h, target_w) torch.LongTensor, remapped to contiguous IDs
    """
    segments_np = segments.cpu().numpy().astype(np.float32)
    resized = cv2.resize(segments_np, (target_w, target_h),
                         interpolation=cv2.INTER_NEAREST)

    # Remap to contiguous IDs
    unique = np.unique(resized)
    label_map = {old: new for new, old in enumerate(unique)}
    h, w = resized.shape
    remapped = np.empty((h, w), dtype=np.int64)
    for y in range(h):
        for x in range(w):
            remapped[y, x] = label_map[resized[y, x]]

    return torch.from_numpy(remapped).long()
