#!/usr/bin/env python3
"""Validate annotation alignment and render image/mask/depth previews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def decode_rle(rle: dict) -> np.ndarray:
    encoded = {
        "size": rle["size"],
        "counts": rle["counts"].encode("ascii") if isinstance(rle["counts"], str) else rle["counts"],
    }
    decoded = mask_utils.decode(encoded)
    if decoded.ndim == 3:
        decoded = decoded[..., 0]
    return decoded.astype(np.uint8)


def load_records(path: Path, limit: int) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
            if len(records) >= limit:
                break
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = args.jsonl.parent
    records = load_records(args.jsonl, args.count)
    report = {
        "records_checked": len(records),
        "mask_shape_matches_image": 0,
        "depth_files_found": 0,
        "visualizations_written": 0,
        "failures": [],
    }

    for index, item in enumerate(records):
        image_id = str(item.get("image_id", index))
        try:
            image_path = Path(item["image_path"])
            image = np.asarray(Image.open(image_path).convert("RGB"))
            height, width = image.shape[:2]

            rle = item.get("mask_rle")
            if not rle:
                raise ValueError("No union mask RLE available")
            mask = decode_rle(rle)
            if mask.shape != (height, width):
                raise ValueError(
                    f"Mask/image alignment mismatch: mask={mask.shape}, image={(height, width)}"
                )
            report["mask_shape_matches_image"] += 1

            mask_tint = np.zeros_like(image)
            mask_tint[..., 0] = 255
            overlay = np.where(mask[..., None] > 0, (0.55 * image + 0.45 * mask_tint), image)
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            depth_path = root / str(item["depth_map_path"])
            if not depth_path.is_file():
                raise FileNotFoundError(f"Depth file missing: {depth_path}")
            with np.load(depth_path) as saved:
                depth = saved["depth"].astype(np.float32)
            if depth.ndim != 2 or not np.isfinite(depth).all():
                raise ValueError(f"Invalid depth array: shape={depth.shape}")
            report["depth_files_found"] += 1

            depth_resized = cv2.resize(depth, (width, height), interpolation=cv2.INTER_CUBIC)
            depth_u8 = np.clip(depth_resized * 255.0, 0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
            depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

            panel = np.concatenate([image, overlay, depth_color], axis=1)
            safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in image_id)[:60]
            output_path = args.output_dir / f"{index:04d}_{safe_name}.jpg"
            if not cv2.imwrite(str(output_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)):
                raise IOError(f"Could not write preview: {output_path}")
            report["visualizations_written"] += 1
        except Exception as exc:
            report["failures"].append({"image_id": image_id, "error": f"{type(exc).__name__}: {exc}"})

    report_path = args.output_dir / "alignment_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
