from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class LossWeights:
    language: float = 1.0
    objectness: float = 1.0
    box_l1: float = 5.0
    box_giou: float = 2.0
    mask_focal: float = 1.0
    mask_dice: float = 1.0
    depth: float = 1.0
    relation: float = 1.0
    phrase: float = 1.0


def language_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected logits [B,L,V] and labels [B,L]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("Language logits and labels must have matching batch/sequence dimensions")
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )


def sigmoid_focal_loss(
    logits: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    targets = targets.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = logits.sigmoid()
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


def dice_loss(logits: Tensor, targets: Tensor, eps: float = 1.0) -> Tensor:
    targets = targets.to(dtype=logits.dtype)
    probabilities = logits.sigmoid()
    dims = tuple(range(2, logits.ndim))
    intersection = (probabilities * targets).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def mask_loss(
    mask_logits: Tensor,
    mask_targets: Tensor,
    valid_entities: Tensor | None = None,
) -> dict[str, Tensor]:
    if mask_logits.ndim != 4 or mask_targets.ndim != 4:
        raise ValueError("Mask tensors must be [B,Q,H,W]")
    if mask_logits.shape[:2] != mask_targets.shape[:2]:
        raise ValueError("Mask batch and query dimensions must match")
    if mask_logits.shape[-2:] != mask_targets.shape[-2:]:
        mask_targets = F.interpolate(
            mask_targets.float(), size=mask_logits.shape[-2:], mode="nearest"
        )
    if valid_entities is not None:
        weights = valid_entities.to(mask_logits.dtype).unsqueeze(-1).unsqueeze(-1)
        normalizer = weights.sum().clamp_min(1.0) * mask_logits.shape[-2] * mask_logits.shape[-1]
        focal = (sigmoid_focal_loss_per_element(mask_logits, mask_targets) * weights).sum() / normalizer
        probabilities = mask_logits.sigmoid()
        targets = mask_targets.to(mask_logits.dtype)
        dims = (2, 3)
        intersection = (probabilities * targets).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
        per_entity_dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        dice = (per_entity_dice * valid_entities.to(per_entity_dice.dtype)).sum() / valid_entities.sum().clamp_min(1)
    else:
        focal = sigmoid_focal_loss(mask_logits, mask_targets)
        dice = dice_loss(mask_logits, mask_targets)
    return {"mask_focal": focal, "mask_dice": dice}


def sigmoid_focal_loss_per_element(
    logits: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    targets = targets.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = logits.sigmoid()
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_t * (1.0 - p_t).pow(gamma) * bce


def box_iou_and_giou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> tuple[Tensor, Tensor]:
    if boxes1.ndim != 2 or boxes2.ndim != 2 or boxes1.shape[-1] != 4 or boxes2.shape[-1] != 4:
        raise ValueError("Boxes must have shape [N,4] and [M,4]")
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(-1)
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(-1)
    union = area1[:, None] + area2[None, :] - intersection
    iou = intersection / union.clamp_min(eps)
    enclosing_tl = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    enclosing_br = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing_area = (enclosing_br - enclosing_tl).clamp_min(0).prod(-1)
    giou = iou - (enclosing_area - union) / enclosing_area.clamp_min(eps)
    return iou, giou


def matched_box_loss(predicted: Tensor, target: Tensor) -> dict[str, Tensor]:
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[-1] != 4:
        raise ValueError("Matched predicted and target boxes must both be [N,4]")
    if predicted.numel() == 0:
        zero = predicted.sum() * 0.0
        return {"box_l1": zero, "box_giou": zero}
    _, giou = box_iou_and_giou(predicted.float(), target.float())
    matched_giou = giou.diagonal()
    return {
        "box_l1": F.l1_loss(predicted, target),
        "box_giou": (1.0 - matched_giou).mean(),
    }


def relative_depth_uncertainty_loss(
    predicted_relative_depth: Tensor,
    log_variance: Tensor,
    target_relative_depth: Tensor,
    valid_mask: Tensor | None = None,
    smoothness_weight: float = 0.05,
) -> dict[str, Tensor]:
    target = target_relative_depth.float()
    if valid_mask is None:
        valid = torch.isfinite(target) & (target > 0)
    else:
        valid = valid_mask.bool()
        if valid.shape != target.shape:
            valid = F.interpolate(valid.float(), size=target.shape[-2:], mode="nearest").bool()
        valid = valid & torch.isfinite(target) & (target > 0)

    safe_target = torch.where(valid, target, torch.ones_like(target))
    if predicted_relative_depth.shape != safe_target.shape:
        safe_target = F.interpolate(
            safe_target, size=predicted_relative_depth.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        valid = F.interpolate(
            valid.float(), size=predicted_relative_depth.shape[-2:], mode="nearest"
        ).bool()
    if log_variance.shape != predicted_relative_depth.shape:
        log_variance = F.interpolate(
            log_variance.float(), size=predicted_relative_depth.shape[-2:],
            mode="bilinear", align_corners=False,
        )

    pred = predicted_relative_depth.float().clamp_min(1e-6).log()
    target_log = safe_target.clamp_min(1e-6).log()
    valid = valid & torch.isfinite(target_log) & torch.isfinite(pred)
    valid_f = valid.to(pred.dtype)
    count = valid_f.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)

    residual = torch.where(valid, pred - target_log, torch.zeros_like(pred))
    mean_residual = residual.sum(dim=(2, 3), keepdim=True) / count
    scale_invariant = torch.where(
        valid, residual - mean_residual, torch.zeros_like(residual)
    )

    variance = log_variance.float().clamp(-8.0, 8.0)
    nll = 0.5 * (torch.exp(-variance) * scale_invariant.square() + variance)
    depth_nll = (nll * valid_f).sum() / valid_f.sum().clamp_min(1.0)

    # Invalid target locations are excluded from the target-edge weighting.
    target_for_smoothness = torch.where(valid, target_log, pred.detach())
    dx = (pred[..., :, 1:] - pred[..., :, :-1]).abs()
    dy = (pred[..., 1:, :] - pred[..., :-1, :]).abs()
    target_dx = (target_for_smoothness[..., :, 1:] - target_for_smoothness[..., :, :-1]).abs()
    target_dy = (target_for_smoothness[..., 1:, :] - target_for_smoothness[..., :-1, :]).abs()
    valid_x = (valid[..., :, 1:] & valid[..., :, :-1]).to(pred.dtype)
    valid_y = (valid[..., 1:, :] & valid[..., :-1, :]).to(pred.dtype)
    smooth_x = (dx * torch.exp(-target_dx) * valid_x).sum() / valid_x.sum().clamp_min(1.0)
    smooth_y = (dy * torch.exp(-target_dy) * valid_y).sum() / valid_y.sum().clamp_min(1.0)
    smoothness = smooth_x + smooth_y

    total = depth_nll + smoothness_weight * smoothness
    return {
        "depth_nll": depth_nll,
        "depth_smoothness": smoothness,
        "depth_total": total,
    }

def relation_classification_loss(logits: Tensor, targets: Tensor) -> Tensor:
    if logits.ndim != 4 or targets.ndim != 3:
        raise ValueError("Expected relation logits [B,Q,Q,R] and targets [B,Q,Q]")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1).long(),
        ignore_index=-100,
    )


def phrase_grounding_loss(entity_phrase_logits: Tensor, targets: Tensor) -> Tensor:
    if entity_phrase_logits.ndim != 3:
        raise ValueError("entity_phrase_logits must be [B,num_entities,num_phrases]")
    if targets.ndim != 2 or targets.shape != (entity_phrase_logits.shape[0], entity_phrase_logits.shape[2]):
        raise ValueError("targets must be [B,num_phrases] with target entity indices")
    return F.cross_entropy(entity_phrase_logits.transpose(1, 2), targets.long(), ignore_index=-100)


def combine_losses(
    components: dict[str, Tensor],
    weights: LossWeights | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    weights = weights or LossWeights()
    mapping = {
        "language": weights.language,
        "objectness": weights.objectness,
        "box_l1": weights.box_l1,
        "box_giou": weights.box_giou,
        "mask_focal": weights.mask_focal,
        "mask_dice": weights.mask_dice,
        "depth_total": weights.depth,
        "relation": weights.relation,
        "phrase": weights.phrase,
    }
    total = None
    weighted = {}
    for name, value in components.items():
        if name not in mapping:
            continue
        contribution = value * mapping[name]
        weighted[f"weighted_{name}"] = contribution.detach()
        total = contribution if total is None else total + contribution
    if total is None:
        raise ValueError("No recognized loss components were supplied")
    weighted["total"] = total.detach()
    return total, weighted
