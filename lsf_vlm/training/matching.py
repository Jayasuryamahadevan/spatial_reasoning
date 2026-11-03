from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from .losses import box_iou_and_giou, sigmoid_focal_loss


@torch.no_grad()
def hungarian_match(
    predicted_boxes: Tensor,
    objectness_logits: Tensor,
    target_boxes: Sequence[Tensor],
    cost_l1: float = 5.0,
    cost_giou: float = 2.0,
    cost_objectness: float = 1.0,
) -> list[tuple[Tensor, Tensor]]:
    """Match entity slots to ground-truth boxes independently for each batch row.

    Args:
        predicted_boxes: normalized xyxy boxes [B,Q,4].
        objectness_logits: objectness logits [B,Q].
        target_boxes: sequence of N_i normalized xyxy tensors [N_i,4].
    Returns:
        Per-image (source_slot_indices, target_indices), both on prediction device.
    """
    if predicted_boxes.ndim != 3 or predicted_boxes.shape[-1] != 4:
        raise ValueError("predicted_boxes must be [B,Q,4]")
    if objectness_logits.shape != predicted_boxes.shape[:2]:
        raise ValueError("objectness_logits must be [B,Q]")
    if len(target_boxes) != predicted_boxes.shape[0]:
        raise ValueError("target_boxes length must equal batch size")

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError("Hungarian matching requires scipy: pip install scipy") from exc

    device = predicted_boxes.device
    matches: list[tuple[Tensor, Tensor]] = []
    for batch_index, targets in enumerate(target_boxes):
        targets = targets.to(device=device, dtype=predicted_boxes.dtype)
        if targets.ndim != 2 or targets.shape[-1] != 4:
            raise ValueError("Each target box tensor must be [N_i,4]")
        if targets.shape[0] == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            matches.append((empty, empty))
            continue

        predicted = predicted_boxes[batch_index].float()
        target = targets.float()
        l1_cost = torch.cdist(predicted, target, p=1)
        _, giou = box_iou_and_giou(predicted, target)
        positive_cost = -objectness_logits[batch_index].float().sigmoid().clamp_min(1e-8).log()
        cost = cost_l1 * l1_cost + cost_giou * (1.0 - giou) + cost_objectness * positive_cost[:, None]
        source, target_index = linear_sum_assignment(cost.cpu().numpy())
        matches.append((
            torch.as_tensor(source, dtype=torch.long, device=device),
            torch.as_tensor(target_index, dtype=torch.long, device=device),
        ))
    return matches


def objectness_focal_loss(
    objectness_logits: Tensor,
    matches: Sequence[tuple[Tensor, Tensor]],
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Supervise all entity slots as positive if matched, otherwise negative."""
    if objectness_logits.ndim != 2:
        raise ValueError("objectness_logits must be [B,Q]")
    if len(matches) != objectness_logits.shape[0]:
        raise ValueError("matches length must equal batch size")
    targets = torch.zeros_like(objectness_logits)
    for batch_index, (source_indices, _) in enumerate(matches):
        if source_indices.numel():
            targets[batch_index, source_indices] = 1.0
    return sigmoid_focal_loss(objectness_logits, targets, alpha=alpha, gamma=gamma)


def matched_entity_box_losses(
    predicted_boxes: Tensor,
    objectness_logits: Tensor,
    target_boxes: Sequence[Tensor],
    cost_l1: float = 5.0,
    cost_giou: float = 2.0,
    cost_objectness: float = 1.0,
) -> tuple[dict[str, Tensor], list[tuple[Tensor, Tensor]]]:
    """Return objectness + matched box losses and the assignment for masks/labels."""
    from .losses import matched_box_loss

    matches = hungarian_match(
        predicted_boxes,
        objectness_logits,
        target_boxes,
        cost_l1=cost_l1,
        cost_giou=cost_giou,
        cost_objectness=cost_objectness,
    )
    device = predicted_boxes.device
    box_l1_terms = []
    box_giou_terms = []
    for batch_index, (source_indices, target_indices) in enumerate(matches):
        if source_indices.numel() == 0:
            continue
        terms = matched_box_loss(
            predicted_boxes[batch_index, source_indices],
            target_boxes[batch_index].to(device=device)[target_indices],
        )
        box_l1_terms.append(terms["box_l1"])
        box_giou_terms.append(terms["box_giou"])

    zero = predicted_boxes.sum() * 0.0
    box_l1 = torch.stack(box_l1_terms).mean() if box_l1_terms else zero
    box_giou = torch.stack(box_giou_terms).mean() if box_giou_terms else zero
    objectness = objectness_focal_loss(objectness_logits, matches)
    return {
        "objectness": objectness,
        "box_l1": box_l1,
        "box_giou": box_giou,
    }, matches
