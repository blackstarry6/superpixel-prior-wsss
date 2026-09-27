"""
Superpixel prototype pooling — pure GPU vectorized.

Computes L2-normalized prototype vectors for each superpixel
by averaging backbone features within that superpixel's region.
"""

import torch
import torch.nn.functional as F


def superpixel_pool(features, segments, min_pixels=10):
    """
    Compute superpixel prototypes from backbone features.

    Args:
        features:  (C, H, W) float32, backbone feature map (e.g. conv5)
        segments:  (H, W) int64, superpixel labels (0..K-1), contiguous
        min_pixels: int, minimum pixels for a superpixel to be considered valid

    Returns:
        prototypes: (K, C) float32, L2-normalized prototype per superpixel
        counts:     (K,) int64, number of pixels in each superpixel
        valid:      (K,) bool, whether superpixel meets min_pixels threshold
    """
    C, H, W = features.shape
    device = features.device

    K = int(segments.max().item()) + 1

    # Ensure segments are on the same device as features
    segments = segments.to(device)

    # Flatten
    feat_flat = features.view(C, -1)    # (C, H*W)
    seg_flat = segments.view(-1).to(torch.int64)  # (H*W)

    # Count pixels per superpixel
    counts = torch.zeros(K, dtype=torch.int64, device=device)
    ones = torch.ones_like(seg_flat, dtype=torch.int64)
    counts.scatter_add_(0, seg_flat, ones)

    # Sum features per superpixel
    feat_sum = torch.zeros(C, K, device=device, dtype=features.dtype)
    seg_expanded = seg_flat.unsqueeze(0).expand(C, -1)  # (C, H*W)
    feat_sum.scatter_add_(1, seg_expanded, feat_flat)

    # Compute prototypes (avoid div-by-zero)
    counts_f = counts.float().clamp(min=1)
    prototypes = feat_sum / counts_f.unsqueeze(0)  # (C, K)

    # L2 normalize each prototype
    prototypes = F.normalize(prototypes, p=2, dim=0)  # (C, K)

    # Valid mask
    valid = counts >= min_pixels

    # Zero out invalid prototypes
    prototypes[:, ~valid] = 0.0

    return prototypes.T, counts, valid  # (K, C), (K,), (K,)


def build_adjacency(segments, valid):
    """
    Build superpixel adjacency matrix from pixel-level connectivity (4-neighbor).

    Args:
        segments: (H, W) int64, superpixel labels
        valid:    (K,) bool, which superpixels are valid

    Returns:
        adj_pairs: (E, 2) int64, unique undirected adjacency pairs (i, j) with i < j
    """
    H, W = segments.shape
    device = segments.device

    # Check right and down neighbors for label transitions
    right = segments[:, 1:]        # (H, W-1)
    left  = segments[:, :-1]
    h_adj = right != left
    h_pairs = torch.stack([
        torch.min(left[h_adj], right[h_adj]),
        torch.max(left[h_adj], right[h_adj]),
    ], dim=1)  # (E_h, 2)

    down  = segments[1:, :]        # (H-1, W)
    up    = segments[:-1, :]
    v_adj = down != up
    v_pairs = torch.stack([
        torch.min(up[v_adj], down[v_adj]),
        torch.max(up[v_adj], down[v_adj]),
    ], dim=1)  # (E_v, 2)

    all_pairs = torch.cat([h_pairs, v_pairs], dim=0)  # (E, 2)
    if all_pairs.shape[0] == 0:
        return all_pairs

    # Deduplicate
    all_pairs = torch.unique(all_pairs, dim=0)

    # Filter: both superpixels must be valid
    mask = valid[all_pairs[:, 0]] & valid[all_pairs[:, 1]]
    adj_pairs = all_pairs[mask]

    return adj_pairs
