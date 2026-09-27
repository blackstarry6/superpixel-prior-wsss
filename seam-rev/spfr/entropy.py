"""
CAM entropy computation for entropy-weighted SPFR.

When CAM shows multi-class confusion at a superpixel (high entropy),
the SPFR foreground pulling weight is reduced to avoid pulling
features of different classes together.
"""

import torch
import torch.nn.functional as F


def compute_superpixel_entropy(cam, segments, K, fg_mask):
    """
    Compute per-superpixel CAM class-distribution entropy.

    Args:
        cam:      (21, H, W) float32, CAM logits (raw, before softmax),
                  channels: [bg, cls_0, cls_1, ..., cls_19]
        segments: (H, W) int64, superpixel labels
        K:        int, number of superpixels
        fg_mask:  (K,) bool, which superpixels are foreground

    Returns:
        entropy: (K,) float32, entropy per superpixel.
                 Background/non-fg superpixels have entropy = 0.0.
                 Range: [0, ln(21)] ≈ [0, 3.04]
    """
    device = cam.device
    entropy = torch.zeros(K, device=device)

    if not fg_mask.any():
        return entropy

    # Softmax over class dimension for each pixel
    cam_soft = F.softmax(cam, dim=0)  # (21, H, W)

    # For each foreground superpixel, compute mean class distribution and its entropy
    for k in range(K):
        if not fg_mask[k]:
            continue
        mask_k = (segments == k)
        if mask_k.sum() == 0:
            continue

        # Mean class distribution in this superpixel
        sp_dist = cam_soft[:, mask_k].mean(dim=1)  # (21,)
        sp_dist = sp_dist.clamp(min=1e-8)
        # Entropy: H = -sum(p * log(p))
        entropy[k] = -(sp_dist * torch.log(sp_dist)).sum()

    return entropy


def entropy_to_weight(entropy, fg_mask):
    """
    Convert entropy to foreground pulling weight.

    weight = 1 / (1 + entropy)

    Interpretation:
      - Low entropy  (single clear class) → weight ≈ 1.0 → strong pull
      - High entropy (multi-class confusion) → weight ≈ 0.2-0.3 → weak pull

    Args:
        entropy: (K,) float32, per-superpixel CAM entropy
        fg_mask: (K,) bool, which superpixels are foreground

    Returns:
        weight: (K,) float32, pulling weight ∈ (0, 1]
    """
    weight = torch.ones_like(entropy)
    weight[fg_mask] = 1.0 / (1.0 + entropy[fg_mask])
    return weight
