# LSF-VLM: Phase 1 teacher distillation

This repository contains a resumable annotation pipeline for a 50K-image CC3M subset.

## Hardware and execution plan

- Select Kaggle GPU T4 x2.
- Launch two Accelerate processes, one process per GPU.
- Each process receives a deterministic half of the image manifest.
- Teachers are loaded sequentially: GroundingDINO-Tiny, SAM ViT-H, then Depth Anything V2 Small.
- Initial batch size is one image per GPU; SAM receives at most eight boxes per image.
- Each stage checkpoints each image to JSONL so a run can resume after interruption.

The two T4 cards have separate 16 GB VRAM pools; they do not form one 32 GB device.

## 1. Prepare a local manifest

Supply a CSV or TSV with these columns:

    image_id,image_path,caption
    cc3m_000001,images/000001.jpg,A dog running across grass.

Image paths may be absolute or relative to the manifest. Images must already be downloaded locally. CC3M metadata contains image URLs, not a local image archive. Use a downloader such as img2dataset to materialize the selected images before annotation.

Example Kaggle manifest:

    /kaggle/input/cc3m-subset/manifest.csv

## 2. Install dependencies

Do not reinstall torch or torchvision in the Kaggle environment. Preserve its CUDA-compatible build.

    pip install -r requirements.txt
    pip install git+https://github.com/facebookresearch/segment-anything.git

## 3. Run on T4 x2

Select GPU T4 x2 under Notebook Settings > Accelerator.

    accelerate launch --multi_gpu --num_processes 2 teachers/teacher_pipeline.py \
      --manifest /kaggle/input/cc3m-subset/manifest.csv \
      --output-dir /kaggle/working/lsf_vlm_distillation \
      --max-samples 50000 \
      --max-detections 8 \
      --stage all

The script downloads the official SAM ViT-H checkpoint into the output directory if it is missing. The checkpoint is approximately 2.5 GB; attaching it as a Kaggle Dataset avoids repeat downloads.

For the first run, use --max-samples 8. Do not launch all 50K images until the smoke test and visual checks pass.

## 4. Output layout

    lsf_vlm_distillation/
      distilled.jsonl
      depth_maps/
      intermediate/
        boxes.rank0.jsonl
        boxes.rank1.jsonl
        masks.rank0.jsonl
        masks.rank1.jsonl
        depth.rank0.jsonl
        depth.rank1.jsonl
      models/
        sam_vit_h_4b8939.pth

The final JSONL includes image_id, text, boxes, labels, box_scores, mask_rle, instances, depth_map_path, depth_representation, and status.

mask_rle is a COCO compressed RLE encoding for the union of all detected instance masks. instances preserves individual box-to-mask correspondence. Depth maps are 256x256 normalized relative-depth arrays stored as float16 NPZ files. They are not metric depth in metres.

## 5. Resume after interruption

Re-run the same command with the same output directory and the same number of processes. Completed image IDs are skipped independently for each stage. Keep the process count fixed at two while resuming.

Run only one stage with --stage boxes, --stage masks, or --stage depth. Rebuild the merged JSONL without rerunning teachers with --stage merge.

## 6. Phase 1 acceptance gate

1. Run eight images.
2. Confirm both GPU processes with nvidia-smi.
3. Confirm all three stage files contain records.
4. Run evaluation/visualize_samples.py against the final JSONL.
5. Inspect 100 images before increasing to 50K.

Record peak allocated/reserved VRAM, images per second, failed-image count, and mask coverage for each stage. Memory figures in the research plan are estimates; measured Kaggle logs are the acceptance evidence.

## Research and licensing notes

- GroundingDINO captions are weak referring expressions, not guaranteed object labels. Review false positives and empty masks.
- CC3M captions and image URLs are noisy. Removed, unavailable, and corrupt images must be filtered.
- Check the terms for source images and each teacher checkpoint before redistribution.
- LLaVA-34B is intentionally excluded from the free T4x2 baseline. Original CC3M captions provide initial text supervision.
