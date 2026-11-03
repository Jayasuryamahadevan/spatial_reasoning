#!/usr/bin/env python3
"""Inspect LFM3-VL feature and hidden-state interfaces before adding LSF heads."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import transformers
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--question", default="Describe the objects and their spatial relationships.")
    parser.add_argument("--model", default="LiquidAI/LFM3-VL-2B")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def tensor_summary(value: Any, prefix: str = "output") -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []

    def visit(item: Any, path: str, depth: int = 0) -> None:
        if depth > 5:
            return
        if torch.is_tensor(item):
            summaries.append({
                "path": path,
                "shape": list(item.shape),
                "dtype": str(item.dtype),
                "device": str(item.device),
            })
        elif isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{path}.{key}", depth + 1)
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]", depth + 1)
        elif hasattr(item, "to_tuple"):
            visit(item.to_tuple(), path, depth + 1)

    visit(value, prefix)
    return summaries


def hookable_module(name: str) -> bool:
    lowered = name.lower()
    terms = ("vision", "visual", "projector", "merger", "patch_embed")
    return any(term in lowered for term in terms)


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    version = transformers.__version__
    print(f"transformers={version}")
    try:
        major, minor = (int(part) for part in version.split(".")[:2])
    except ValueError:
        major, minor = 0, 0
    if (major, minor) < (5, 21):
        raise RuntimeError(
            "The current LFM3-VL model card requires Transformers >=5.21.0. "
            "Use an isolated LFM3 environment; do not upgrade the Phase 1 teacher environment in place."
        )

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available; this run will use CPU and may be very slow.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats(device)
        print(f"cuda_device={torch.cuda.get_device_name(device)}")
        print(f"cuda_total_memory_gib={torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f}")

    print(f"loading_model={args.model}")
    model_load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map={"": 0} if device.type == "cuda" else "cpu",
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    ).eval()
    model_load_seconds = time.perf_counter() - model_load_start

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"model_load_seconds={model_load_seconds:.3f}")
    print(f"parameters_total={total_params:,} ({total_params / 1e9:.4f}B)")
    print(f"parameters_trainable_before_adaptation={trainable_params:,}")

    module_hits = []
    hook_handles = []

    def make_hook(module_name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            module_hits.append({
                "module": module_name,
                "tensors": tensor_summary(output, module_name),
            })
        return hook

    # Hook only named vision/projector modules. These are observation hooks;
    # they do not modify activations or model parameters.
    for name, module in model.named_modules():
        if name and hookable_module(name):
            if len(hook_handles) >= 80:
                break
            hook_handles.append(module.register_forward_hook(make_hook(name)))

    image = Image.open(args.image).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": args.question},
        ],
    }]
    model_inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_summary = tensor_summary(model_inputs, "inputs")
    # device_map={"": 0} places the model on GPU 0; move tensor inputs there.
    if device.type == "cuda":
        model_inputs = model_inputs.to(device)

    print("input_tensors:")
    print(json.dumps(input_summary, indent=2))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    forward_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_start

    output_summary = tensor_summary(outputs, "outputs")
    hidden_states = getattr(outputs, "hidden_states", None)
    hidden_summary = tensor_summary(hidden_states, "hidden_states") if hidden_states is not None else []

    for handle in hook_handles:
        handle.remove()

    report: dict[str, Any] = {
        "model": args.model,
        "transformers_version": version,
        "image": str(args.image.resolve()),
        "image_size": list(image.size),
        "question": args.question,
        "model_load_seconds": model_load_seconds,
        "forward_seconds": forward_seconds,
        "parameters_total": total_params,
        "parameters_trainable_before_adaptation": trainable_params,
        "input_tensors": input_summary,
        "output_tensors": output_summary,
        "hidden_state_tensors": hidden_summary,
        "vision_module_hooks": module_hits,
    }

    if device.type == "cuda":
        report["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
        report["cuda_peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 1024**3

    print("\nFORWARD OUTPUT TENSORS")
    print(json.dumps(output_summary, indent=2))
    print("\nLANGUAGE HIDDEN STATES")
    print(json.dumps(hidden_summary, indent=2))
    print("\nVISION / PROJECTOR HOOKS")
    print(json.dumps(module_hits, indent=2))

    if not hidden_states:
        raise RuntimeError(
            "The model did not return hidden_states. Inspect the model forward signature "
            "and its custom output class before designing the language-query tap."
        )
    if not module_hits:
        print(
            "WARNING: No named vision/projector modules were caught by the current hook filter. "
            "Inspect model.named_modules() and update hookable_module() before proceeding."
        )

    if device.type == "cuda":
        print(f"\npeak_allocated_gib={report['cuda_peak_allocated_gib']:.3f}")
        print(f"peak_reserved_gib={report['cuda_peak_reserved_gib']:.3f}")
    print(f"forward_seconds={forward_seconds:.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"report_saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
