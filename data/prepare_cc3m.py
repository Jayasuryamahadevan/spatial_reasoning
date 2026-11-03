#!/usr/bin/env python3
"""Sample CC3M metadata and build the local image_path manifest after img2dataset."""

from __future__ import annotations

import argparse
import csv
import gzip
import random
from pathlib import Path

import pyarrow.parquet as pq


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open(
        "r", encoding="utf-8", newline=""
    )


def sample_metadata(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    reservoir: list[tuple[str, str]] = []
    seen = 0
    with open_text(args.source_tsv) as handle:
        first = handle.readline()
        first_fields = first.rstrip("\n").split("\t")
        has_header = {"caption", "url"}.issubset({value.strip().lower() for value in first_fields})

        def parse_fields(fields: list[str]) -> tuple[str, str] | None:
            if has_header:
                header = [value.strip().lower() for value in first_fields]
                mapping = dict(zip(header, fields))
                caption, url = mapping.get("caption", "").strip(), mapping.get("url", "").strip()
            else:
                if len(fields) < 2:
                    return None
                if args.source_order == "caption_url":
                    caption, url = fields[0].strip(), fields[1].strip()
                else:
                    url, caption = fields[0].strip(), fields[1].strip()
            if not caption or not url or not url.startswith(("http://", "https://")):
                return None
            return url, caption

        reader = csv.reader(handle, delimiter="\t")
        if not has_header:
            parsed = parse_fields(first_fields)
            if parsed:
                seen += 1
                reservoir.append(parsed)

        for fields in reader:
            parsed = parse_fields(fields)
            if not parsed:
                continue
            seen += 1
            if len(reservoir) < args.sample_size:
                reservoir.append(parsed)
            else:
                replacement = rng.randrange(seen)
                if replacement < args.sample_size:
                    reservoir[replacement] = parsed

    if not reservoir:
        raise RuntimeError("No valid CC3M URL/caption rows were found.")
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["url", "caption"])
        writer.writerows(reservoir)
    print(f"Scanned {seen:,} valid metadata rows; wrote {len(reservoir):,} sampled pairs to {args.output_tsv}")


def build_manifest(args: argparse.Namespace) -> None:
    output_root = args.images_dir.resolve()
    parquet_files = sorted(output_root.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No img2dataset shard Parquet metadata found directly under {output_root}"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    written = missing = failed = 0
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "image_path", "caption"])
        writer.writeheader()
        for parquet_path in parquet_files:
            table = pq.read_table(parquet_path, columns=["key", "caption", "status"])
            for row in table.to_pylist():
                key = str(row.get("key") or "")
                caption = str(row.get("caption") or "").strip()
                status = str(row.get("status") or "").lower()
                if status not in {"success", "ok", "200"}:
                    failed += 1
                    continue
                if len(key) < 6 or not caption:
                    failed += 1
                    continue
                shard = key[:5]
                image_path = output_root / shard / f"{key}.jpg"
                if not image_path.is_file():
                    missing += 1
                    continue
                writer.writerow({
                    "image_id": key,
                    "image_path": str(image_path),
                    "caption": caption,
                })
                written += 1

    print(
        f"Manifest: {args.output_csv}\n"
        f"Usable images: {written:,}\n"
        f"Failed downloads/invalid metadata: {failed:,}\n"
        f"Missing image files: {missing:,}"
    )
    if written == 0:
        raise RuntimeError(
            "No usable images were found. Check the img2dataset version, parquet key format, "
            "and output directory layout."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="Reservoir-sample the raw CC3M TSV")
    sample.add_argument("--source-tsv", required=True, type=Path)
    sample.add_argument("--output-tsv", required=True, type=Path)
    sample.add_argument("--sample-size", type=int, default=50_000)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--source-order", choices=["caption_url", "url_caption"], default="caption_url")

    manifest = subparsers.add_parser("manifest", help="Build the CSV from img2dataset files + shard metadata")
    manifest.add_argument("--images-dir", required=True, type=Path)
    manifest.add_argument("--output-csv", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "sample":
        if args.sample_size < 1:
            raise ValueError("--sample-size must be positive")
        sample_metadata(args)
    else:
        build_manifest(args)


if __name__ == "__main__":
    main()
