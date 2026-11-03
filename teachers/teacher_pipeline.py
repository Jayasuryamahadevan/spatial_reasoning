#!/usr/bin/env python3
"""Resumable two-GPU teacher annotation pipeline for LSF-VLM."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import random
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

LOG = logging.getLogger("lsf_distillation")
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
DEFAULT_GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, help="CSV/TSV: image_id,image_path,caption")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--image-column", default="image_path")
    p.add_argument("--id-column", default="image_id")
    p.add_argument("--caption-column", default="caption")
    p.add_argument("--max-samples", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grounding-model", default=DEFAULT_GROUNDING_MODEL)
    p.add_argument("--depth-model", default=DEFAULT_DEPTH_MODEL)
    p.add_argument("--sam-checkpoint", default=None)
    p.add_argument("--box-threshold", type=float, default=0.30)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--max-detections", type=int, default=8)
    p.add_argument("--depth-size", type=int, default=256)
    p.add_argument("--stage", choices=["all", "boxes", "masks", "depth", "merge"], default="all")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def setup_logging(level: str, rank: int) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=f"%(asctime)s | rank={rank} | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def read_manifest(path: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        required = {args.id_column, args.image_column, args.caption_column}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        rows, seen = [], set()
        for row in reader:
            image_id = str(row[args.id_column]).strip()
            image_path = str(row[args.image_column]).strip()
            caption = str(row[args.caption_column]).strip()
            if not image_id or not image_path or not caption or image_id in seen:
                continue
            seen.add(image_id)
            resolved = Path(image_path)
            if not resolved.is_absolute():
                resolved = (path.parent / resolved).resolve()
            rows.append({"image_id": image_id, "image_path": str(resolved), "text": caption[:1000]})
    if not rows:
        raise ValueError("No usable rows found in manifest.")
    if args.max_samples > 0 and len(rows) > args.max_samples:
        rows = random.Random(args.seed).sample(rows, args.max_samples)
    rows.sort(key=lambda item: item["image_id"])
    return rows


def safe_stem(image_id: str) -> str:
    digest = hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:16]
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_id).strip("._-")[:40]
    return f"{readable or 'image'}_{digest}"


def rank_stage_path(output_dir: Path, stage: str, rank: int) -> Path:
    return output_dir / "intermediate" / f"{stage}.rank{rank}.jsonl"


def read_jsonl_map(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        LOG.warning("Ignoring an incomplete checkpoint line in %s", path)
                        continue
                    records[str(item["image_id"])] = item
    return records


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def stage_todo(rows: list[dict[str, str]], stage_file: Path) -> list[dict[str, str]]:
    done = read_jsonl_map(stage_file)
    # Retry failed/upstream-error records on the next invocation; only status=ok is final.
    completed_ids = {image_id for image_id, item in done.items() if item.get("status") == "ok"}
    return [row for row in rows if row["image_id"] not in completed_ids]


def load_rgb(image_path: str) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_peak(stage: str, device: torch.device) -> None:
    torch.cuda.synchronize(device)
    gib = 1024**3
    LOG.info(
        "%s peak VRAM: allocated=%.2f GiB reserved=%.2f GiB",
        stage,
        torch.cuda.max_memory_allocated(device) / gib,
        torch.cuda.max_memory_reserved(device) / gib,
    )


def run_boxes(rows: list[dict[str, str]], output_dir: Path, acc: Accelerator, args: argparse.Namespace) -> None:
    rank = acc.process_index
    out_path = rank_stage_path(output_dir, "boxes", rank)
    todo = stage_todo(rows, out_path)
    if not todo:
        LOG.info("GroundingDINO stage already complete for this rank.")
        return

    LOG.info("Loading GroundingDINO %s; pending=%d", args.grounding_model, len(todo))
    torch.cuda.reset_peak_memory_stats(acc.device)
    processor = AutoProcessor.from_pretrained(args.grounding_model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.grounding_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(acc.device).eval()

    for row in tqdm(todo, desc=f"boxes/rank{rank}", disable=not acc.is_local_main_process):
        try:
            image = load_rgb(row["image_path"])
            width, height = image.size
            # The CC3M caption is a weak referring expression, not a verified class label.
            inputs = processor(images=image, text=[[row["text"][:256]]], return_tensors="pt")
            inputs = {
                key: value.to(acc.device, dtype=(torch.float16 if key == "pixel_values" else value.dtype))
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                outputs = model(**inputs)
            result = processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], threshold=args.box_threshold,
                text_threshold=args.text_threshold, target_sizes=[(height, width)]
            )[0]
            boxes = result["boxes"].detach().float().cpu().numpy()
            scores = result["scores"].detach().float().cpu().numpy()
            # Transformers >=4.51 returns integer IDs from "labels";
            # "text_labels" preserves the grounded phrases.
            labels = result.get("text_labels") or result.get("labels", [])
            order = np.argsort(scores)[::-1][:args.max_detections]
            kept_boxes, kept_scores, kept_labels = [], [], []
            for index in order:
                x1, y1, x2, y2 = boxes[index].tolist()
                x1, x2 = float(np.clip(x1, 0, width)), float(np.clip(x2, 0, width))
                y1, y2 = float(np.clip(y1, 0, height)), float(np.clip(y2, 0, height))
                if x2 <= x1 or y2 <= y1:
                    continue
                kept_boxes.append([x1, y1, x2, y2])
                kept_scores.append(float(scores[index]))
                kept_labels.append(str(labels[index]))
            append_jsonl(out_path, {
                "image_id": row["image_id"], "status": "ok", "boxes": kept_boxes,
                "scores": kept_scores, "labels": kept_labels, "image_size": [height, width],
            })
        except Exception as exc:
            LOG.exception("GroundingDINO failed for %s", row["image_id"])
            append_jsonl(out_path, {
                "image_id": row["image_id"], "status": "error",
                "error": f"{type(exc).__name__}: {exc}", "boxes": [], "scores": [], "labels": [],
            })
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                release_cuda()

    log_peak("GroundingDINO", acc.device)
    del model, processor
    release_cuda()


def ensure_sam_checkpoint(checkpoint: Path) -> None:
    if checkpoint.exists() and checkpoint.stat().st_size > 1_000_000_000:
        return
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Downloading SAM ViT-H checkpoint to %s", checkpoint)
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".part")
    urllib.request.urlretrieve(SAM_URL, temporary)
    if temporary.stat().st_size < 1_000_000_000:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("SAM checkpoint download is incomplete.")
    temporary.replace(checkpoint)


def encode_rle(binary_mask: np.ndarray) -> dict[str, Any]:
    from pycocotools import mask as mask_utils
    encoded = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(x) for x in encoded["size"]], "counts": counts}


def run_masks(rows: list[dict[str, str]], output_dir: Path, acc: Accelerator, args: argparse.Namespace) -> None:
    rank = acc.process_index
    out_path = rank_stage_path(output_dir, "masks", rank)
    todo = stage_todo(rows, out_path)
    if not todo:
        LOG.info("SAM stage already complete for this rank.")
        return
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise RuntimeError(
            "Install SAM: pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from exc

    checkpoint = Path(args.sam_checkpoint) if args.sam_checkpoint else output_dir / "models" / "sam_vit_h_4b8939.pth"
    ensure_sam_checkpoint(checkpoint)
    LOG.info("Loading SAM ViT-H; pending=%d", len(todo))
    torch.cuda.reset_peak_memory_stats(acc.device)
    sam = sam_model_registry["vit_h"](checkpoint=str(checkpoint)).to(acc.device).eval()
    predictor = SamPredictor(sam)
    box_records = read_jsonl_map(rank_stage_path(output_dir, "boxes", rank))

    for row in tqdm(todo, desc=f"masks/rank{rank}", disable=not acc.is_local_main_process):
        try:
            box_record = box_records.get(row["image_id"])
            if box_record is None:
                raise RuntimeError("Missing GroundingDINO checkpoint for this image.")
            if box_record.get("status") != "ok":
                append_jsonl(out_path, {
                    "image_id": row["image_id"], "status": "upstream_error",
                    "error": box_record.get("error", "GroundingDINO failed"),
                    "mask_rle": None, "instances": [],
                })
                continue

            image = np.asarray(load_rgb(row["image_path"]))
            height, width = image.shape[:2]
            with torch.inference_mode(), acc.autocast():
                predictor.set_image(image)
            boxes = box_record.get("boxes", [])
            instances, binary_masks = [], []
            if boxes:
                box_tensor = torch.as_tensor(boxes, dtype=torch.float32, device=acc.device)
                transformed = predictor.transform.apply_boxes_torch(box_tensor, (height, width))
                with torch.inference_mode(), acc.autocast():
                    masks, iou_scores, _ = predictor.predict_torch(
                        point_coords=None, point_labels=None, boxes=transformed,
                        multimask_output=False, return_logits=False,
                    )
                masks_np = masks[:, 0].to(torch.uint8).cpu().numpy()
                iou_np = iou_scores.reshape(-1).float().cpu().numpy()
                for index, mask in enumerate(masks_np):
                    binary = (mask > 0).astype(np.uint8)
                    binary_masks.append(binary)
                    instances.append({
                        "box": boxes[index],
                        "label": box_record["labels"][index],
                        "detector_score": box_record["scores"][index],
                        "sam_iou_score": float(iou_np[index]),
                        "mask_rle": encode_rle(binary),
                    })
            union = np.maximum.reduce(binary_masks).astype(np.uint8) if binary_masks else np.zeros((height, width), np.uint8)
            append_jsonl(out_path, {
                "image_id": row["image_id"], "status": "ok",
                "mask_rle": encode_rle(union), "instances": instances,
                "image_size": [height, width],
            })
            predictor.reset_image()
        except Exception as exc:
            LOG.exception("SAM failed for %s", row["image_id"])
            append_jsonl(out_path, {
                "image_id": row["image_id"], "status": "error",
                "error": f"{type(exc).__name__}: {exc}", "mask_rle": None, "instances": [],
            })
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                release_cuda()

    log_peak("SAM ViT-H", acc.device)
    del predictor, sam
    release_cuda()


def run_depth(rows: list[dict[str, str]], output_dir: Path, acc: Accelerator, args: argparse.Namespace) -> None:
    rank = acc.process_index
    out_path = rank_stage_path(output_dir, "depth", rank)
    todo = stage_todo(rows, out_path)
    if not todo:
        LOG.info("Depth stage already complete for this rank.")
        return

    LOG.info("Loading Depth Anything V2 Small %s; pending=%d", args.depth_model, len(todo))
    torch.cuda.reset_peak_memory_stats(acc.device)
    processor = AutoImageProcessor.from_pretrained(args.depth_model)
    model = AutoModelForDepthEstimation.from_pretrained(
        args.depth_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(acc.device).eval()

    for row in tqdm(todo, desc=f"depth/rank{rank}", disable=not acc.is_local_main_process):
        try:
            image = load_rgb(row["image_path"])
            inputs = processor(images=image, return_tensors="pt")
            inputs = {
                key: value.to(acc.device, dtype=(torch.float16 if torch.is_floating_point(value) else value.dtype))
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                outputs = model(**inputs)
            post = processor.post_process_depth_estimation(
                outputs, target_sizes=[(image.height, image.width)]
            )
            depth = post[0]["predicted_depth"].float()
            lo, hi = torch.quantile(depth, 0.02), torch.quantile(depth, 0.98)
            depth = ((depth - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
            depth = torch.nn.functional.interpolate(
                depth[None, None], size=(args.depth_size, args.depth_size),
                mode="bilinear", align_corners=False,
            )[0, 0]
            depth_array = depth.cpu().numpy().astype(np.float16)

            relative_path = Path("depth_maps") / f"{safe_stem(row['image_id'])}.npz"
            destination = output_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".npz.part")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle, depth=depth_array,
                    original_size=np.asarray([image.height, image.width], dtype=np.int32),
                    representation=np.asarray("relative_normalized"),
                )
            temporary.replace(destination)
            append_jsonl(out_path, {
                "image_id": row["image_id"], "status": "ok",
                "depth_map_path": relative_path.as_posix(),
                "depth_representation": "relative_normalized",
                "depth_size": [args.depth_size, args.depth_size],
            })
        except Exception as exc:
            LOG.exception("Depth failed for %s", row["image_id"])
            append_jsonl(out_path, {
                "image_id": row["image_id"], "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "depth_map_path": None, "depth_representation": "relative_normalized",
            })
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                release_cuda()

    log_peak("Depth Anything V2 Small", acc.device)
    del model, processor
    release_cuda()


def merge_outputs(rows: list[dict[str, str]], output_dir: Path, world_size: int) -> Path:
    maps = {}
    for stage in ("boxes", "masks", "depth"):
        merged = {}
        for rank in range(world_size):
            merged.update(read_jsonl_map(rank_stage_path(output_dir, stage, rank)))
        maps[stage] = merged

    final_path = output_dir / "distilled.jsonl"
    temporary = final_path.with_suffix(".jsonl.part")
    written = missing = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            image_id = row["image_id"]
            boxes, masks, depth = (maps[name].get(image_id) for name in ("boxes", "masks", "depth"))
            if not boxes or not masks or not depth:
                missing += 1
                continue
            errors = {
                name: record.get("error", record.get("status"))
                for name, record in (("boxes", boxes), ("masks", masks), ("depth", depth))
                if record.get("status") != "ok"
            }
            item = {
                "image_id": image_id,
                "image_path": row["image_path"],
                "text": row["text"],
                "boxes": boxes.get("boxes", []),
                "labels": boxes.get("labels", []),
                "box_scores": boxes.get("scores", []),
                "mask_rle": masks.get("mask_rle"),
                "instances": masks.get("instances", []),
                "depth_map_path": depth.get("depth_map_path"),
                "depth_representation": depth.get("depth_representation", "relative_normalized"),
                "status": "ok" if not errors else "partial",
            }
            if errors:
                item["errors"] = errors
            handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
            written += 1
    temporary.replace(final_path)
    LOG.info("Merged %d records; %d missing at least one stage. Output: %s", written, missing, final_path)
    return final_path


def main() -> None:
    args = parse_args()
    acc = Accelerator(mixed_precision="fp16")
    setup_logging(args.log_level, acc.process_index)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Select GPU T4 x2 in Kaggle Notebook settings.")

    manifest = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest, args)
    shard = rows[acc.process_index::acc.num_processes]

    LOG.info(
        "images=%d shard=%d rank=%d/%d device=%s gpu=%s",
        len(rows), len(shard), acc.process_index, acc.num_processes,
        acc.device, torch.cuda.get_device_name(acc.device),
    )

    if args.stage in {"all", "boxes"}:
        run_boxes(shard, output_dir, acc, args)
        acc.wait_for_everyone()
    if args.stage in {"all", "masks"}:
        run_masks(shard, output_dir, acc, args)
        acc.wait_for_everyone()
    if args.stage in {"all", "depth"}:
        run_depth(shard, output_dir, acc, args)
        acc.wait_for_everyone()
    if args.stage in {"all", "merge"}:
        acc.wait_for_everyone()
        if acc.is_main_process:
            merge_outputs(rows, output_dir, acc.num_processes)
        acc.wait_for_everyone()


if __name__ == "__main__":
    main()
