"""
SPFR: Superpixel Prototype Feature Regularization.

A plug-and-play module for WSSS classification network training.
Imposes spatial-semantic structure on backbone features via
superpixel-guided prototype contrastive learning.

Usage:
    from spfr import spfr_loss
    from spfr.superpixel import generate_superpixels, resize_superpixels

    # In training loop:
    segments, _ = generate_superpixels(img_tensor)
    segments_ds = resize_superpixels(segments, feat_h, feat_w)
    loss_spfr, l_intra, l_inter, n_pairs = spfr_loss(
        features, cam.detach(), segments_ds, labels,
        lambda_intra=0.1, lambda_inter=0.05
    )
    total_loss = loss_cls + loss_er + loss_ecr + loss_spfr
"""

from .loss import spfr_loss
from .superpixel import generate_superpixels, resize_superpixels
from .pooling import superpixel_pool, build_adjacency
from .entropy import compute_superpixel_entropy, entropy_to_weight

__all__ = [
    'spfr_loss',
    'generate_superpixels',
    'resize_superpixels',
    'superpixel_pool',
    'build_adjacency',
    'compute_superpixel_entropy',
    'entropy_to_weight',
]
