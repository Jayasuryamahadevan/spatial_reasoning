import json

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

from data.unified_dataset import UnifiedSpatialJsonlDataset


def _rle(mask: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {
        "size": [int(value) for value in encoded["size"]],
        "counts": encoded["counts"].decode("ascii"),
    }


def test_unified_dataset_loads_partial_teacher_labels(tmp_path):
    image = Image.new("RGB", (32, 24), color=(30, 40, 50))
    image_path = tmp_path / "sample.jpg"
    image.save(image_path)

    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[4:18, 5:20] = 1
    depth_path = tmp_path / "depth.npz"
    np.savez_compressed(
        depth_path,
        depth=np.full((16, 16), 0.5, dtype=np.float16),
        representation=np.asarray("relative_normalized"),
    )

    record = {
        "image_id": "sample-1",
        "image_path": str(image_path),
        "text": "A cup on a table.",
        "boxes": [[5, 4, 20, 18], [0, 0, 32, 24]],
        "instances": [
            {
                "box": [5, 4, 20, 18],
                "label": "cup",
                "sam_iou_score": 0.9,
                "mask_rle": _rle(mask),
            },
            {
                "box": [0, 0, 32, 24],
                "label": "table",
                "sam_iou_score": 0.2,
                "mask_rle": _rle(np.ones((24, 32), dtype=np.uint8)),
            },
        ],
        "depth_map_path": str(depth_path),
        "depth_representation": "relative_normalized",
        "status": "partial",
    }
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = UnifiedSpatialJsonlDataset(
        jsonl,
        min_sam_iou=0.5,
        min_mask_pixels=4,
    )
    sample = dataset[0]

    assert sample["image"].size == (32, 24)
    assert sample["boxes"].shape == (2, 4)
    assert torch.allclose(sample["boxes"][0], torch.tensor([5 / 32, 4 / 24, 20 / 32, 18 / 24]))
    assert sample["mask_valid"].tolist() == [True, False]
    assert sample["masks"].shape == (2, 24, 32)
    assert sample["depth"].shape == (1, 16, 16)
    assert sample["labels"] == ["cup", "table"]
