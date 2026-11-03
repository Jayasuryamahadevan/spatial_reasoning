#!/usr/bin/env python3
"""Run one end-to-end LFM3 + LSF spatial forward pass and verify single vision execution."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

from lsf_vlm.modeling import UnifiedLSFVLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--question", default="Describe the scene and the spatial relationships between its objects.")
    parser.add_argument("--model", default="LiquidAI/LFM3-VL-2B")
    parser.add_argument("--output", type=Path, default=Path("lsf_smoke_report.json"))
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test expects a CUDA GPU. Use Kaggle T4 or a CUDA workstation.")

    device = torch.device("cuda:0")
    dtype = dtype_from_name(args.dtype)
    torch.cuda.reset_peak_memory_stats(device)

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Transformers: {__import__('transformers').__version__}")
    print("Loading LFM3 processor and base model...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = UnifiedLSFVLM.from_pretrained(
        model_id=args.model,
        torch_dtype=dtype,
        device_map=None,
        freeze_backbone=True,
    ).eval()

    image = Image.open(args.image).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": args.question},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    required = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    missing = [key for key in required if key not in inputs]
    if missing:
        raise RuntimeError(f"LFM3 processor did not return required fields: {missing}")

    # grid_thw is [T, patch_grid_height, patch_grid_width]. The processor
    # preserves aspect ratio and resizes dimensions to multiples of patch*merge.
    grid = inputs["image_grid_thw"]
    if grid.ndim != 2 or grid.shape[0] != 1 or int(grid[0, 0]) != 1:
        raise RuntimeError(f"Expected one still image; received image_grid_thw={grid.tolist()}")
    patch_size = int(model.backbone.config.vision_config.patch_size)
    resized_height = int(grid[0, 1]) * patch_size
    resized_width = int(grid[0, 2]) * patch_size
    resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    rgb = torch.from_numpy(
        __import__("numpy").array(resized, dtype="float32", copy=True)
    ).permute(2, 0, 1).unsqueeze(0) / 255.0

    model_inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    rgb = rgb.to(device=device, dtype=dtype)

    vision_call_count = {"count": 0}

    def count_vision_calls(_module: torch.nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
        vision_call_count["count"] += 1

    handle = model.backbone.model.vision_tower.register_forward_hook(count_vision_calls)
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate_answer(
            tokenizer=processor.tokenizer,
            max_new_tokens=24,
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            pixel_values=model_inputs["pixel_values"],
            image_grid_thw=model_inputs["image_grid_thw"],
            rgb=rgb,
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    handle.remove()

    tensor_shapes = {
        key: list(value.shape)
        for key, value in outputs.items()
        if torch.is_tensor(value)
    }
    report = {
        "image": str(args.image.resolve()),
        "original_image_size": list(image.size),
        "processed_rgb_size": [resized_width, resized_height],
        "grid_thw": grid.tolist(),
        "vision_tower_calls": vision_call_count["count"],
        "forward_and_decode_seconds": elapsed,
        "answer_text": outputs.get("answer_text", ""),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        "parameter_report": model.parameter_report(),
        "tensor_shapes": tensor_shapes,
    }

    if vision_call_count["count"] != 1:
        raise AssertionError(
            f"Expected exactly one LFM3 vision-tower execution, got {vision_call_count['count']}"
        )
    if outputs["spatial_tokens"].shape[1] != 64:
        raise AssertionError("Spatial token bridge did not return exactly 64 tokens")
    if not torch.isfinite(outputs["mask_logits"]).all():
        raise AssertionError("Mask logits contain non-finite values")
    if not torch.isfinite(outputs["relative_depth"]).all():
        raise AssertionError("Relative depth contains non-finite values")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
