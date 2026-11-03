from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _resolve_path(value: str | None, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _decode_coco_rle(rle: dict[str, Any]) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError("COCO RLE decoding requires pycocotools: pip install pycocotools") from exc

    encoded = dict(rle)
    counts = encoded.get("counts")
    if isinstance(counts, str):
        encoded["counts"] = counts.encode("ascii")
    decoded = mask_utils.decode(encoded)
    if decoded.ndim == 3:
        decoded = decoded[..., 0]
    return (decoded > 0).astype(np.uint8)


def _pixel_boxes_to_normalized(
    boxes: list[list[float]],
    image_width: int,
    image_height: int,
    normalized: bool = False,
) -> torch.Tensor:
    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32)
    result = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    if not normalized:
        scale = torch.tensor(
            [image_width, image_height, image_width, image_height],
            dtype=torch.float32,
        )
        result = result / scale
    result = result.clamp(0.0, 1.0)
    x1y1 = torch.minimum(result[:, :2], result[:, 2:])
    x2y2 = torch.maximum(result[:, :2], result[:, 2:])
    return torch.cat((x1y1, x2y2), dim=-1)


class UnifiedSpatialJsonlDataset(Dataset):
    """Read LSF's unified JSONL.

    Supported fields include image_path, text, boxes, instances[].mask_rle,
    depth_map_path, and optional relation_triplets. The weak CC3M teacher
    output is accepted directly. Missing task labels remain missing; they
    are never converted into negative targets.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        data_root: str | Path | None = None,
        image_root: str | Path | None = None,
        max_samples: int | None = None,
        min_sam_iou: float = 0.5,
        min_mask_pixels: int = 16,
        max_objects: int = 32,
    ) -> None:
        self.jsonl_path = Path(jsonl_path).resolve()
        if not self.jsonl_path.is_file():
            raise FileNotFoundError(self.jsonl_path)
        self.data_root = Path(data_root).resolve() if data_root else self.jsonl_path.parent
        self.image_root = Path(image_root).resolve() if image_root else self.data_root
        self.min_sam_iou = float(min_sam_iou)
        self.min_mask_pixels = int(min_mask_pixels)
        self.max_objects = int(max_objects)
        self.rows: list[dict[str, Any]] = []

        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {self.jsonl_path}:{line_number}") from exc
                if row.get("status") not in {"ok", "partial", None}:
                    continue
                image_path = _resolve_path(row.get("image_path"), self.data_root)
                if image_path is None or not image_path.is_file():
                    stored_path = Path(str(row.get("image_path", ""))).expanduser()
                    candidates = []
                    if stored_path.name:
                        candidates.append(self.image_root / stored_path.name)
                    if len(stored_path.parts) >= 2:
                        candidates.append(self.image_root / stored_path.parts[-2] / stored_path.name)
                    if "cc3m_images" in stored_path.parts:
                        offset = stored_path.parts.index("cc3m_images")
                        candidates.append(self.image_root.joinpath(*stored_path.parts[offset + 1:]))
                    image_path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
                if image_path is None or not image_path.is_file():
                    continue
                row["_resolved_image_path"] = str(image_path)
                self.rows.append(row)
                if max_samples and len(self.rows) >= max_samples:
                    break

        if not self.rows:
            raise ValueError(f"No usable image records found in {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = Path(row["_resolved_image_path"])
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        normalized = bool(row.get("boxes_normalized", False))
        raw_boxes = row.get("boxes_xyxy_norm") if normalized else row.get("boxes")
        if raw_boxes is None:
            raw_boxes = [instance.get("box") for instance in row.get("instances", []) if instance.get("box") is not None]
        boxes = _pixel_boxes_to_normalized(raw_boxes or [], width, height, normalized=normalized)
        if boxes.shape[0] > self.max_objects:
            boxes = boxes[: self.max_objects]

        instances = row.get("instances") or []
        masks = torch.zeros((boxes.shape[0], height, width), dtype=torch.float32)
        mask_valid = torch.zeros((boxes.shape[0],), dtype=torch.bool)
        labels: list[str] = []

        row_labels = row.get("labels") or []
        for object_index in range(boxes.shape[0]):
            instance = instances[object_index] if object_index < len(instances) else {}
            fallback_label = row_labels[object_index] if object_index < len(row_labels) else ""
            labels.append(str(instance.get("label") or fallback_label))
            rle = instance.get("mask_rle")
            if not isinstance(rle, dict):
                continue
            sam_iou = float(instance.get("sam_iou_score", 1.0))
            if sam_iou < self.min_sam_iou:
                continue
            try:
                decoded = _decode_coco_rle(rle)
            except Exception:
                continue
            if decoded.shape != (height, width):
                continue
            area = int(decoded.sum())
            if area < self.min_mask_pixels or area >= int(height * width * 0.995):
                continue
            masks[object_index] = torch.from_numpy(decoded.astype(np.float32))
            mask_valid[object_index] = True

        depth = None
        depth_representation = row.get("depth_representation")
        depth_path = _resolve_path(row.get("depth_map_path"), self.data_root)
        if depth_path is not None and depth_path.is_file():
            with np.load(depth_path, allow_pickle=False) as archive:
                key = "depth" if "depth" in archive else archive.files[0]
                depth_array = np.asarray(archive[key], dtype=np.float32)
            if depth_array.ndim == 2:
                depth = torch.from_numpy(depth_array[None, ...])
                depth = torch.where(
                    torch.isfinite(depth) & (depth > 0),
                    depth,
                    torch.zeros_like(depth),
                )

        relation_matrix = None
        relation_triplets = row.get("relation_triplets")
        if relation_triplets:
            from lsf_vlm.modeling.spatial_core import RELATION_LABELS

            relation_to_id = {name: idx for idx, name in enumerate(RELATION_LABELS)}
            relation_matrix = torch.full(
                (boxes.shape[0], boxes.shape[0]), -100, dtype=torch.long
            )
            for relation in relation_triplets:
                subject = int(relation["subject"])
                object_ = int(relation["object"])
                predicate = str(relation["predicate"])
                if (
                    0 <= subject < boxes.shape[0]
                    and 0 <= object_ < boxes.shape[0]
                    and subject != object_
                    and predicate in relation_to_id
                ):
                    relation_matrix[subject, object_] = relation_to_id[predicate]

        return {
            "image_id": str(row.get("image_id", index)),
            "image": image,
            "image_path": str(image_path),
            "caption": str(row.get("text", row.get("caption", ""))),
            "boxes": boxes,
            "masks": masks,
            "mask_valid": mask_valid,
            "labels": labels,
            "depth": depth,
            "depth_representation": depth_representation,
            "relations": relation_matrix,
        }


def single_item_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError("The initial LFM3 adapter requires batch_size=1 per process.")
    return items[0]
