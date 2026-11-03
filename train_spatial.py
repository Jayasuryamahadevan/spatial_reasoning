#!/usr/bin/env python3
"""Train LSF spatial heads on unified JSONL with frozen LFM3-VL features.

This is the stage-1 spatial warmup. The LFM3 backbone is frozen and its
language path runs without autograd. Spatial outputs train jointly from one
vision pass. Optional --train-language-prefix enables language gradients only
through the inserted continuous spatial prefix; it does not unfreeze LFM3.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from data.unified_dataset import UnifiedSpatialJsonlDataset, single_item_collate
from lsf_vlm.modeling import UnifiedLSFVLM
from lsf_vlm.training.losses import (
    LossWeights,
    language_cross_entropy,
    mask_loss,
    phrase_grounding_loss,
    relation_classification_loss,
    relative_depth_uncertainty_loss,
)
from lsf_vlm.training.matching import (
    build_matched_mask_targets,
    matched_entity_box_losses,
    remap_phrase_targets_to_entity_slots,
    remap_relation_targets_to_entity_slots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--model", default="LiquidAI/LFM3-VL-2B")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lsf_spatial"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--min-sam-iou", type=float, default=0.0)
    parser.add_argument("--min-mask-pixels", type=int, default=16)
    parser.add_argument("--max-objects", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-language-prefix", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=0)
    return parser.parse_args()


def prepare_processor_inputs(
    processor: Any,
    image: Image.Image,
    question: str,
    device: torch.device,
    dtype: torch.dtype,
    patch_size: int,
    answer: str | None = None,
    phrases: list[tuple[str, int]] | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None, tuple[int, int]]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = prompt["input_ids"]
    attention_mask = prompt["attention_mask"]
    labels = None

    if answer:
        answer_ids = processor.tokenizer(
            answer,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
        eos_id = processor.tokenizer.eos_token_id
        if eos_id is not None:
            answer_ids = torch.cat(
                (answer_ids, torch.tensor([[eos_id]], dtype=answer_ids.dtype)),
                dim=1,
            )
        prompt_length = input_ids.shape[1]
        input_ids = torch.cat((input_ids, answer_ids), dim=1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(answer_ids, dtype=attention_mask.dtype)),
            dim=1,
        )
        labels = torch.full_like(input_ids, -100)
        labels[:, prompt_length:] = answer_ids

    processor_inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in prompt.items()
    }
    processor_inputs["input_ids"] = input_ids.to(device)
    processor_inputs["attention_mask"] = attention_mask.to(device)

    grid = prompt["image_grid_thw"]
    if grid.ndim != 2 or grid.shape[0] != 1 or int(grid[0, 0]) != 1:
        raise RuntimeError(f"Expected one still image; got image_grid_thw={grid.tolist()}")

    resized_h = int(grid[0, 1]) * patch_size
    resized_w = int(grid[0, 2]) * patch_size
    resized = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
    rgb = torch.from_numpy(
        np.array(resized, dtype=np.float32, copy=True)
    ).permute(2, 0, 1).unsqueeze(0) / 255.0
    rgb = rgb.to(device=device, dtype=dtype)
    question_mask = torch.zeros_like(input_ids, dtype=torch.long)
    # The user question is everything after the image block up to the original
    # generation prompt boundary. It excludes the assistant target.
    image_end_id = int(processor.tokenizer.convert_tokens_to_ids("<|vision_end|>"))
    end_positions = torch.nonzero(input_ids[0] == image_end_id, as_tuple=False).flatten()
    if not end_positions.numel():
        raise RuntimeError("The processor chat template did not emit <|vision_end|>.")
    start = int(end_positions[0].item()) + 1
    stop = prompt["input_ids"].shape[1]
    question_mask[:, start:stop] = 1

    phrase_token_masks = None
    phrase_targets = None
    phrases = [(phrase, int(target_index)) for phrase, target_index in (phrases or []) if phrase.strip()]
    if phrases:
        sequence = input_ids[0].tolist()
        search_from = start
        masks = []
        matched_targets = []
        for phrase, target_index in phrases:
            token_variants = [
                processor.tokenizer(text, add_special_tokens=False)["input_ids"]
                for text in (" " + phrase, phrase)
            ]
            token_variants = [tokens for tokens in token_variants if tokens]
            found = None
            matched_phrase_ids = None
            for phrase_ids in token_variants:
                last_start = min(stop, len(sequence)) - len(phrase_ids)
                for candidate_start in range(search_from, last_start + 1):
                    if sequence[candidate_start:candidate_start + len(phrase_ids)] == phrase_ids:
                        found = candidate_start
                        matched_phrase_ids = phrase_ids
                        break
                if found is not None:
                    break
            if found is None:
                for phrase_ids in token_variants:
                    last_start = min(stop, len(sequence)) - len(phrase_ids)
                    for candidate_start in range(start, last_start + 1):
                        if sequence[candidate_start:candidate_start + len(phrase_ids)] == phrase_ids:
                            found = candidate_start
                            matched_phrase_ids = phrase_ids
                            break
                    if found is not None:
                        break
            if found is None or matched_phrase_ids is None:
                continue
            mask = torch.zeros_like(input_ids, dtype=torch.long)
            mask[0, found:found + len(matched_phrase_ids)] = 1
            masks.append(mask)
            matched_targets.append(target_index)
            search_from = found + len(matched_phrase_ids)
        if masks:
            phrase_token_masks = torch.stack(masks, dim=1).to(device)
            phrase_targets = torch.tensor(
                [matched_targets], dtype=torch.long, device=device
            )

    model_inputs = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "pixel_values": prompt["pixel_values"].to(device),
        "image_grid_thw": grid.to(device),
        "rgb": rgb,
        "question_mask": question_mask.to(device),
    }
    if phrase_token_masks is not None:
        model_inputs["phrase_token_masks"] = phrase_token_masks
    if labels is not None:
        model_inputs["labels"] = labels.to(device)
    return (
        model_inputs,
        question_mask.to(device),
        labels.to(device) if labels is not None else None,
        (resized_h, resized_w),
        phrase_targets,
    )


def compute_losses(
    outputs: dict[str, Any],
    sample: dict[str, Any],
    device: torch.device,
    train_language_prefix: bool,
    loss_weights: LossWeights,
    phrase_targets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    components: dict[str, torch.Tensor] = {}
    target_boxes = sample["boxes"].to(device=device, dtype=torch.float32)
    matches = None

    if target_boxes.shape[0] > 0:
        box_components, matches = matched_entity_box_losses(
            outputs["boxes_xyxy_norm"],
            outputs["objectness_logits"],
            [target_boxes],
        )
        components.update(box_components)

        target_masks = sample["masks"].to(device=device, dtype=torch.float32)
        target_mask_valid = sample["mask_valid"].to(device=device, dtype=torch.bool)
        if target_masks.shape[0] == target_boxes.shape[0] and target_mask_valid.any():
            matched_masks, valid_entities = build_matched_mask_targets(
                matches,
                [target_masks],
                num_queries=outputs["mask_logits"].shape[1],
                output_size=outputs["mask_logits"].shape[-2:],
                device=device,
                target_mask_valid=[target_mask_valid],
            )
            mask_components = mask_loss(
                outputs["mask_logits"],
                matched_masks,
                valid_entities=valid_entities,
            )
            components.update(mask_components)

        if sample["relations"] is not None:
            relation_targets = remap_relation_targets_to_entity_slots(
                matches,
                [sample["relations"]],
                num_queries=outputs["relation_logits"].shape[1],
                device=device,
            )
            components["relation"] = relation_classification_loss(
                outputs["relation_logits"],
                relation_targets,
            )

        if phrase_targets is not None and "entity_phrase_logits" in outputs:
            phrase_slots = remap_phrase_targets_to_entity_slots(
                matches,
                phrase_targets,
                num_queries=outputs["entity_phrase_logits"].shape[1],
            )
            components["phrase"] = phrase_grounding_loss(
                outputs["entity_phrase_logits"],
                phrase_slots,
            )

    if sample["depth"] is not None:
        target_depth = sample["depth"].unsqueeze(0).to(device=device, dtype=torch.float32)
        depth_components = relative_depth_uncertainty_loss(
            outputs["relative_depth"],
            outputs["depth_log_variance"],
            target_depth,
        )
        components["depth_total"] = depth_components["depth_total"]

    if train_language_prefix and outputs.get("labels") is not None:
        components["language"] = language_cross_entropy(
            outputs["language_logits"].float(),
            outputs["labels"],
        )

    if not components:
        raise RuntimeError(f"Sample {sample['image_id']} has no usable task labels.")

    total, weighted = combine_spatial_losses(components, loss_weights)
    return total, {**components, **weighted}


def combine_spatial_losses(
    components: dict[str, torch.Tensor],
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
    metrics: dict[str, torch.Tensor] = {}
    for key, value in components.items():
        if key not in mapping:
            continue
        weighted = value * mapping[key]
        total = weighted if total is None else total + weighted
        metrics[f"weighted_{key}"] = weighted.detach()
    if total is None:
        raise RuntimeError("No recognized LSF spatial losses were computed.")
    metrics["total"] = total.detach()
    return total, metrics


def save_checkpoint(
    accelerator: Accelerator,
    model: UnifiedLSFVLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: Path,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "spatial": unwrapped.spatial.state_dict(),
            "spatial_to_language": unwrapped.spatial_to_language.state_dict(),
            "spatial_type_embedding": unwrapped.spatial_type_embedding.detach().cpu(),
            "layer_fusion_logits": unwrapped.layer_fusion_logits.detach().cpu(),
            "step": step,
            "optimizer": optimizer.state_dict(),
            "parameter_report": unwrapped.parameter_report(),
        }
        torch.save(state, output_dir / f"lsf_spatial_step_{step:08d}.pt")
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("epochs and gradient-accumulation-steps must be positive")
    if args.train_language_prefix is False:
        language_prefix_grad = False
    else:
        language_prefix_grad = True

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(args.seed)

    dataset = UnifiedSpatialJsonlDataset(
        args.train_jsonl,
        data_root=args.data_root,
        image_root=args.image_root,
        max_samples=args.max_samples or None,
        min_sam_iou=args.min_sam_iou,
        min_mask_pixels=args.min_mask_pixels,
        max_objects=args.max_objects,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = UnifiedLSFVLM.from_pretrained(
        model_id=args.model,
        torch_dtype=torch.float16,
        device_map={"": accelerator.local_process_index},
        freeze_backbone=True,
    )
    model.freeze_backbone = True
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)

    if args.gradient_checkpointing and hasattr(model.backbone.model.language_model, "gradient_checkpointing_enable"):
        model.backbone.model.language_model.gradient_checkpointing_enable()

    # Stage 1 only updates the spatial core. Prefix projection is added in
    # stage 2 when language_path_grad is enabled.
    trainable_modules = [model.spatial]
    if args.train_language_prefix:
        trainable_modules += [model.spatial_to_language]
        model.spatial_type_embedding.requires_grad_(True)
    else:
        model.spatial_to_language.requires_grad_(False)
        model.spatial_type_embedding.requires_grad_(False)

    trainable_parameters = [
        parameter
        for module in trainable_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if args.train_language_prefix:
        trainable_parameters += [model.spatial_type_embedding]
    if not trainable_parameters:
        raise RuntimeError("No trainable LSF parameters were found.")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=single_item_collate,
        num_workers=0,
        pin_memory=True,
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()
    accelerator.unwrap_model(model).backbone.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.spatial.load_state_dict(checkpoint["spatial"])
        unwrapped.spatial_to_language.load_state_dict(checkpoint["spatial_to_language"])
        unwrapped.spatial_type_embedding.data.copy_(checkpoint["spatial_type_embedding"].to(unwrapped.spatial_type_embedding.device))
        unwrapped.layer_fusion_logits.data.copy_(checkpoint["layer_fusion_logits"].to(unwrapped.layer_fusion_logits.device))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint.get("step", 0))

    loss_weights = LossWeights()
    optimizer.zero_grad(set_to_none=True)
    running: dict[str, float] = {}
    micro_step = 0
    started = time.perf_counter()

    for epoch in range(args.epochs):
        for sample in loader:
            phrase_pairs = [
                (label, object_index)
                for object_index, label in enumerate(sample["labels"])
                if label
            ]
            phrase_labels = [label for label, _ in phrase_pairs]
            if args.train_language_prefix:
                question = (
                    "Describe the image, identify the objects, and explain their spatial arrangement. "
                    + (f"Candidate object phrases: {', '.join(phrase_labels)}." if phrase_labels else "")
                )
                answer = sample["caption"] or "The image contains objects arranged in a scene."
            else:
                question = (
                    "Identify the objects in this image and estimate their masks and relative depth. "
                    + (f"Object phrases: {', '.join(phrase_labels)}." if phrase_labels else "")
                )
                answer = None

            inputs, question_mask, labels, _, phrase_targets = prepare_processor_inputs(
                processor,
                sample["image"],
                question,
                accelerator.device,
                torch.float16,
                patch_size=int(model.backbone.config.vision_config.patch_size),
                answer=answer,
                phrases=phrase_pairs,
            )
            with accelerator.accumulate(model):
                outputs = model(
                    **inputs,
                    language_path_grad=language_prefix_grad,
                    insert_spatial_prefix=True,
                    compute_language_logits=args.train_language_prefix,
                    use_cache=False,
                )
                loss, metrics = compute_losses(
                    outputs,
                    sample,
                    accelerator.device,
                    train_language_prefix=args.train_language_prefix,
                    loss_weights=loss_weights,
                    phrase_targets=phrase_targets,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

            micro_step += 1
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + float(value.detach().float().cpu())

            if global_step and global_step % args.log_every == 0 and accelerator.sync_gradients:
                if accelerator.is_main_process:
                    averages = {key: value / max(1, micro_step) for key, value in running.items()}
                    print(json.dumps({
                        "epoch": epoch,
                        "step": global_step,
                        "seconds": round(time.perf_counter() - started, 2),
                        "losses": averages,
                    }))
                running.clear()
                micro_step = 0

            if args.save_every > 0 and global_step > 0 and global_step % args.save_every == 0 and accelerator.sync_gradients:
                save_checkpoint(
                    accelerator,
                    model,
                    optimizer,
                    global_step,
                    args.output_dir,
                )

            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    save_checkpoint(accelerator, model, optimizer, global_step, args.output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    main()
