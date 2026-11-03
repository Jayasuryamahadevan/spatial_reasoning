import torch

from lsf_vlm.modeling.spatial_core import (
    LSFSpatialCore,
    SpatialCoreConfig,
    find_vision_end_insertion_index,
    insert_soft_spatial_tokens,
)
from lsf_vlm.training.losses import (
    box_iou_and_giou,
    mask_loss,
    matched_box_loss,
    phrase_grounding_loss,
    relation_classification_loss,
    relative_depth_uncertainty_loss,
)


def make_core() -> LSFSpatialCore:
    config = SpatialCoreConfig(
        vision_dim=1024,
        language_dim=2048,
        spatial_dim=256,
        num_spatial_latents=64,
        num_entities=32,
        num_heads=8,
        perceiver_layers=2,
        entity_layers=2,
        pixel_dim=128,
        num_relations=10,
    )
    return LSFSpatialCore(config)


def test_unified_spatial_shapes_and_finite_outputs() -> None:
    torch.manual_seed(7)
    model = make_core().eval()
    vision = torch.randn(1, 1024, 8, 10)
    rgb = torch.rand(1, 3, 64, 80)
    language = torch.randn(1, 2048)
    phrases = torch.randn(1, 2, 2048)

    with torch.no_grad():
        output = model(vision, rgb, language, phrase_contexts=phrases)

    assert output["spatial_tokens"].shape == (1, 64, 256)
    assert output["entity_tokens"].shape == (1, 32, 256)
    assert output["boxes_xyxy_norm"].shape == (1, 32, 4)
    assert output["objectness_logits"].shape == (1, 32)
    assert output["mask_logits"].shape == (1, 32, 16, 20)
    assert output["relative_inverse_depth"].shape == (1, 1, 64, 80)
    assert output["depth_log_variance"].shape == (1, 1, 64, 80)
    assert output["relation_logits"].shape == (1, 32, 32, 10)
    assert output["entity_phrase_logits"].shape == (1, 32, 2)

    for key in (
        "spatial_tokens",
        "entity_tokens",
        "boxes_xyxy_norm",
        "objectness_logits",
        "mask_logits",
        "relative_inverse_depth",
        "depth_log_variance",
        "relation_logits",
        "entity_phrase_logits",
    ):
        assert torch.isfinite(output[key]).all(), key

    boxes = output["boxes_xyxy_norm"]
    assert ((boxes >= 0.0) & (boxes <= 1.0)).all()
    assert (boxes[..., 0] <= boxes[..., 2]).all()
    assert (boxes[..., 1] <= boxes[..., 3]).all()
    assert (output["relative_inverse_depth"] > 0).all()


def test_spatial_losses_backpropagate_into_shared_core() -> None:
    torch.manual_seed(11)
    model = make_core().train()
    vision = torch.randn(1, 1024, 8, 8)
    rgb = torch.rand(1, 3, 64, 64)
    language = torch.randn(1, 2048)
    output = model(vision, rgb, language)

    target_masks = torch.randint(0, 2, (1, 32, 64, 64)).float()
    masks = mask_loss(output["mask_logits"], target_masks)
    target_depth = torch.rand(1, 1, 64, 64) + 0.1
    depth = relative_depth_uncertainty_loss(
        output["relative_inverse_depth"],
        output["depth_log_variance"],
        target_depth,
    )
    target_boxes = torch.rand(5, 4)
    target_boxes[:, 2:] = torch.maximum(target_boxes[:, 2:], target_boxes[:, :2] + 0.05).clamp_max(1.0)
    boxes = matched_box_loss(
        output["boxes_xyxy_norm"][0, :5],
        target_boxes,
    )

    loss = masks["mask_focal"] + masks["mask_dice"] + depth["depth_total"] + boxes["box_l1"] + boxes["box_giou"]
    loss.backward()

    assert model.spatial_bridge.latent_queries.grad is not None
    assert model.spatial_bridge.latent_queries.grad.abs().sum() > 0
    assert model.pixel_decoder.mask_features[0].weight.grad is not None
    assert model.depth_head.net[-1].weight.grad is not None
    assert model.entity_decoder.box_head[-1].weight.grad is not None


def test_soft_spatial_tokens_are_inserted_without_changing_image_placeholders() -> None:
    ids = torch.tensor([[1, 10, 99, 11, 20, 21, 2]])
    embeds = torch.randn(1, 7, 16)
    mask = torch.ones(1, 7, dtype=torch.long)
    spatial = torch.randn(1, 64, 16)
    insert_at = find_vision_end_insertion_index(ids, vision_end_token_id=11)

    new_ids, new_embeds, new_mask = insert_soft_spatial_tokens(
        ids,
        embeds,
        mask,
        spatial,
        insertion_index=insert_at,
        placeholder_token_id=0,
    )

    assert insert_at == 4
    assert new_ids.shape == (1, 71)
    assert new_embeds.shape == (1, 71, 16)
    assert new_mask.shape == (1, 71)
    assert torch.equal(new_ids[:, :4], ids[:, :4])
    assert torch.equal(new_ids[:, 68:], ids[:, 4:])
    assert torch.equal(new_embeds[:, 4:68], spatial)
    assert new_mask.sum().item() == 71


def test_relation_and_phrase_losses_are_finite() -> None:
    relation_logits = torch.randn(1, 4, 4, 9, requires_grad=True)
    relation_targets = torch.full((1, 4, 4), -100, dtype=torch.long)
    relation_targets[0, 0, 1] = 1
    relation_targets[0, 1, 0] = 2
    rel_loss = relation_classification_loss(relation_logits, relation_targets)

    phrase_logits = torch.randn(1, 4, 2, requires_grad=True)
    phrase_targets = torch.tensor([[0, 3]])
    phrase_loss = phrase_grounding_loss(phrase_logits, phrase_targets)
    (rel_loss + phrase_loss).backward()

    assert torch.isfinite(rel_loss)
    assert torch.isfinite(phrase_loss)
    assert relation_logits.grad is not None
    assert phrase_logits.grad is not None


def test_box_iou_and_giou_are_bounded() -> None:
    boxes1 = torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.0, 0.0, 1.0, 1.0]])
    boxes2 = torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]])
    iou, giou = box_iou_and_giou(boxes1, boxes2)
    assert torch.all((iou >= 0) & (iou <= 1))
    assert torch.all((giou >= -1) & (giou <= 1))
