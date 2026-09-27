"""
SPFR (Superpixel Prototype Feature Regularization) Loss.

Core components:
  L_intra:  within-superpixel feature compactness
  L_inter:  between-superpixel prototype contrast (entropy-weighted FG/BG)

Total: L_spfr = lambda_intra * L_intra + lambda_inter * L_inter
"""

import torch
import torch.nn.functional as F

from .pooling import superpixel_pool, build_adjacency
from .entropy import compute_superpixel_entropy, entropy_to_weight


def spfr_loss(features, cam, segments, labels,
              lambda_intra=0.1, lambda_inter=0.05,
              margin=1.0, threshold=0.3, min_pixels=10):
    """
    Compute SPFR loss for a batch of images.

    Args:
        features:      (B, C, H, W) backbone feature map (conv5: C=1024)
        cam:           (B, 21, H, W) CAM logits (detached, for pseudo-labeling)
        segments:      (B, H, W) int64, superpixel labels
        labels:        (B, 20) float, image-level multi-label ground truth
        lambda_intra:  float, weight for L_intra
        lambda_inter:  float, weight for L_inter
        margin:        float, margin for foreground-background separation
        threshold:     float, CAM confidence threshold for FG/BG determination
        min_pixels:    int, minimum pixels per superpixel

    Returns:
        total_loss:      scalar
        loss_intra_val:  scalar (for logging)
        loss_inter_val:  scalar (for logging)
        num_valid_pairs: int (for logging)
    """
    B = features.size(0)
    device = features.device

    total_intra = torch.tensor(0.0, device=device)
    total_inter = torch.tensor(0.0, device=device)
    total_pairs = 0

    for b in range(B):
        feat_b = features[b]       # (C, H_f, W_f), e.g. (1024, 56, 56)
        cam_b = cam[b].detach()    # (21, H_c, W_c), e.g. (21, 448, 448) — full resolution
        seg_b = segments[b]        # (H_f, W_f), same spatial size as features
        label_b = labels[b]        # (20,) float

        # Resize CAM to match feature/segment spatial size
        if cam_b.shape[-2:] != seg_b.shape:
            cam_b = F.interpolate(
                cam_b.unsqueeze(0),
                size=seg_b.shape,
                mode='bilinear',
                align_corners=True,
            ).squeeze(0)

        K = int(seg_b.max().item()) + 1
        if K < 2:
            continue

        # ── 1. Superpixel prototypes ──────────────────────────
        prototypes, counts, valid = superpixel_pool(feat_b, seg_b, min_pixels=min_pixels)
        # prototypes: (K, C), valid: (K,)

        if valid.sum() < 2:
            continue

        # ── 2. FG/BG pseudo labels (per superpixel) ────────────
        fg_mask, bg_mask = _assign_fg_bg(
            cam_b, seg_b, K, labels=label_b,
            threshold=threshold
        )

        # ── 3. L_intra: within-superpixel compactness ──────────
        loss_intra_b = _compute_intra_loss(feat_b, seg_b, prototypes, valid, counts)
        total_intra += loss_intra_b

        # ── 4. Build adjacency ─────────────────────────────────
        adj_pairs = build_adjacency(seg_b, valid)  # (E, 2)
        if adj_pairs.shape[0] == 0:
            continue

        # ── 5. CAM entropy for foreground superpixels ──────────
        sp_entropy = compute_superpixel_entropy(cam_b, seg_b, K, fg_mask)
        fg_pull_weight = entropy_to_weight(sp_entropy, fg_mask)
        # fg_pull_weight[k] ∈ (0, 1] — high for low-entropy FG regions

        # ── 6. L_inter: prototype contrast with entropy weighting ──
        loss_inter_b, n_pairs = _compute_inter_loss(
            prototypes, adj_pairs, fg_mask, bg_mask,
            fg_pull_weight, margin=margin
        )
        total_inter += loss_inter_b
        total_pairs += n_pairs

    # ── Aggregate ──────────────────────────────────────────────
    if total_pairs == 0:
        return (torch.tensor(0.0, device=device, requires_grad=True),
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
                0)

    loss_intra_val = total_intra / max(B, 1)
    loss_inter_val = total_inter / max(total_pairs, 1)
    total_loss = lambda_intra * loss_intra_val + lambda_inter * loss_inter_val

    return total_loss, loss_intra_val.detach(), loss_inter_val.detach(), total_pairs


# ═══════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════

def _assign_fg_bg(cam, segments, K, labels, threshold=0.3):
    """
    Assign FG/BG pseudo labels to each superpixel.

    FG: any foreground class CAM mean > threshold and > BG mean + 0.1
    BG: BG CAM mean > threshold and > max FG mean + 0.1
    Otherwise: unassigned (neither FG nor BG)

    Args:
        cam:     (21, H, W) CAM logits
        segments:(H, W) superpixel labels
        K:       number of superpixels
        labels:  (20,) image-level multi-label (1 = class present)

    Returns:
        fg_mask: (K,) bool
        bg_mask: (K,) bool
    """
    device = cam.device

    # Find which foreground classes are present in this image
    present_classes = (labels > 0.5).nonzero(as_tuple=True)[0]  # indices 0..19

    fg_mask = torch.zeros(K, dtype=torch.bool, device=device)
    bg_mask = torch.zeros(K, dtype=torch.bool, device=device)

    for k in range(K):
        sp_mask_k = (segments == k)
        n_px = sp_mask_k.sum()
        if n_px == 0:
            continue

        # Background CAM mean
        bg_cam = cam[0, sp_mask_k]  # channel 0 = background
        bg_mean = bg_cam.mean()

        # Max foreground CAM mean among present classes
        fg_best_mean = 0.0
        for c in present_classes:
            fg_c = cam[c + 1, sp_mask_k]  # channels 1..20
            fg_best_mean = max(fg_best_mean, fg_c.mean().item())

        # FG/BG determination
        if fg_best_mean > threshold and fg_best_mean > bg_mean + 0.1:
            fg_mask[k] = True
        elif bg_mean > threshold and bg_mean > fg_best_mean + 0.1:
            bg_mask[k] = True
        # else: neither (ambiguous / ignore)

    return fg_mask, bg_mask


def _compute_intra_loss(features, segments, prototypes, valid, counts):
    """
    L_intra: encourage pixels within each superpixel to be close to its prototype.

    Uses cosine similarity (via dot product after L2 normalization of features).

    Args:
        features:   (C, H, W) backbone features
        segments:   (H, W) superpixel labels
        prototypes: (K, C) L2-normalized prototype vectors
        valid:      (K,) bool
        counts:     (K,) int64

    Returns:
        scalar loss
    """
    C, H, W = features.shape
    device = features.device

    # L2-normalize pixel features
    feat_norm = F.normalize(features, p=2, dim=0)  # (C, H, W)

    # For each valid superpixel, compute mean(1 - cos(feat_pixel, prototype))
    valid_indices = valid.nonzero(as_tuple=True)[0]
    if len(valid_indices) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    total_loss = torch.tensor(0.0, device=device)
    total_px = 0

    for k in valid_indices:
        k = k.item()
        sp_mask = (segments == k)
        n_px = sp_mask.sum().item()
        if n_px == 0:
            continue

        # Pixel features in this superpixel
        feat_k = feat_norm[:, sp_mask]          # (C, n_px)
        proto_k = prototypes[k:k+1].T           # (C, 1)

        # Cosine similarity: cos = feat^T * proto (both L2-normed)
        cos_sim = (feat_k * proto_k).sum(dim=0)  # (n_px,)
        cos_sim = cos_sim.clamp(-1.0, 1.0)

        total_loss += (1.0 - cos_sim).sum()
        total_px += n_px

    if total_px == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / total_px


def _compute_inter_loss(prototypes, adj_pairs, fg_mask, bg_mask,
                        fg_pull_weight, margin=1.0):
    """
    L_inter: entropy-weighted prototype contrast.

    FG-FG adjacent → pull (entropy-weighted)
    FG-BG adjacent → push (full strength)
    BG-BG adjacent → ignore

    Args:
        prototypes:     (K, C) L2-normalized
        adj_pairs:      (E, 2) adjacency index pairs
        fg_mask:        (K,) bool
        bg_mask:        (K,) bool
        fg_pull_weight: (K,) float, entropy-based weight for FG pulling
        margin:         float, margin for push loss

    Returns:
        loss_inter: scalar
        n_pairs:    number of contributing pairs
    """
    device = prototypes.device

    if adj_pairs.shape[0] == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0

    i_idx = adj_pairs[:, 0]
    j_idx = adj_pairs[:, 1]

    proto_i = prototypes[i_idx]  # (E, C)
    proto_j = prototypes[j_idx]  # (E, C)

    # Cosine distance: ||proto_i - proto_j|| (both L2-normed → range [0, 2])
    dist = torch.norm(proto_i - proto_j, p=2, dim=1)  # (E,)

    # Pair types
    fg_fg = fg_mask[i_idx] & fg_mask[j_idx]
    fg_bg = (fg_mask[i_idx] & bg_mask[j_idx]) | (bg_mask[i_idx] & fg_mask[j_idx])

    # Entropy weights for FG-FG pairs
    pair_weight = torch.min(fg_pull_weight[i_idx], fg_pull_weight[j_idx])

    loss = torch.tensor(0.0, device=device)
    n_pairs = 0

    # FG-FG: entropy-weighted pull
    if fg_fg.any():
        pull_loss = (dist[fg_fg] ** 2) * pair_weight[fg_fg]
        loss = loss + pull_loss.sum()
        n_pairs += fg_fg.sum().item()

    # FG-BG: full-strength push
    if fg_bg.any():
        push_loss = torch.clamp(margin - dist[fg_bg], min=0) ** 2
        loss = loss + push_loss.sum()
        n_pairs += fg_bg.sum().item()

    return loss, n_pairs
