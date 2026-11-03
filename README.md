# LSF-VLM: Unified Edge Spatial Reasoning

## Architecture direction

The current architecture is specified in [docs/LSF_VLM_ARCHITECTURE.md](docs/LSF_VLM_ARCHITECTURE.md). The model uses LFM3-VL-2B as its pretrained vision-language backbone, a shared 64-token spatial workspace, language-conditioned task queries, and compact box/mask/depth/relation heads. Teacher models are for offline distillation only; they are not part of deployed inference.

Before building the spatial heads, run the backbone probe:

    pip install -r requirements-lfm3.txt
    python experiments/lfm3_probe.py \
      --image /path/to/test.jpg \
      --question "Where is the red cup relative to the laptop?" \
      --output /tmp/lfm3_probe.json

The integrated one-image gate tests the LFM3 vision tower + 64 continuous spatial tokens + entity, mask, depth, and relation heads in one forward path. It asserts that the LFM3 vision tower executes exactly once:

    python experiments/lsf_one_image_smoke.py \
      --image /path/to/test.jpg \
      --question "Where is the red cup relative to the laptop?" \
      --output /tmp/lsf_smoke_report.json

Run the spatial-core unit tests:

    pytest -q tests/test_spatial_core.py

Neither the integration gate nor the unit tests have been run on Kaggle yet. Record actual output, latency, and peak memory before starting the 50K teacher annotation job.

The LFM3 environment is intentionally separate from the teacher environment. The existing teacher pipeline pins Transformers below 5, while the current LFM3 model card documents Transformers 5.21.0 or newer. Do not upgrade the teacher environment in place.

## Phase 1 teacher distillation (legacy preparation workflow)


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

    /kaggle/working/cc3m_images/manifest.csv

### Create the 50K CC3M image subset

Download the official CC3M metadata TSV first. The original CC3M TSV is commonly distributed without a header and ordered as caption then URL; the helper also detects a header row containing caption and url.

    pip install img2dataset pyarrow

    python data/prepare_cc3m.py sample \
      --source-tsv /kaggle/input/cc3m-metadata/Train_GCC-training.tsv \
      --output-tsv /kaggle/working/cc3m_50k.tsv \
      --sample-size 50000 \
      --seed 42

Download the selected images. img2dataset writes image files and per-shard Parquet metadata; failed or unavailable URLs are recorded and filtered when the manifest is built.

    img2dataset \
      --url_list /kaggle/working/cc3m_50k.tsv \
      --input_format tsv \
      --url_col url \
      --caption_col caption \
      --output_format files \
      --output_folder /kaggle/working/cc3m_images \
      --processes_count 4 \
      --thread_count 16 \
      --image_size 512 \
      --resize_mode keep_ratio_largest \
      --retries 2

Build the manifest consumed by the teacher pipeline:

    python data/prepare_cc3m.py manifest \
      --images-dir /kaggle/working/cc3m_images \
      --output-csv /kaggle/working/cc3m_images/manifest.csv

Check the printed usable-image count. It may be lower than 50,000 because source URLs can fail or be removed. The annotation run processes all valid rows in the manifest up to --max-samples.

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
Kaggle sessions can end before the full annotation job finishes. The per-image checkpoints are resumable only while the output directory is retained. Before ending a session, export the output directory as a Kaggle Dataset or preserve it as a notebook output, then attach that saved output in the next session and continue with the same output directory and two-process configuration.

Run only one stage with --stage boxes, --stage masks, or --stage depth. Rebuild the merged JSONL without rerunning teachers with --stage merge.

## 6. Phase 1 acceptance gate

1. Run eight images.
2. Confirm both GPU processes with nvidia-smi.
3. Confirm all three stage files contain records.
4. Run the alignment preview on 100 records:

       python evaluation/visualize_samples.py --jsonl /kaggle/working/lsf_vlm_distillation/distilled.jsonl --output-dir /kaggle/working/lsf_vlm_distillation/previews --count 100
5. Inspect 100 images before increasing to 50K.

Record peak allocated/reserved VRAM, images per second, failed-image count, and mask coverage for each stage. Memory figures in the research plan are estimates; measured Kaggle logs are the acceptance evidence.

## Research and licensing notes

- GroundingDINO captions are weak referring expressions, not guaranteed object labels. Review false positives and empty masks.
- CC3M captions and image URLs are noisy. Removed, unavailable, and corrupt images must be filtered.
- Check the terms for source images and each teacher checkpoint before redistribution.
- LLaVA-34B is intentionally excluded from the free T4x2 baseline. Original CC3M captions provide initial text supervision.
- Architecture gate before Phase 2: Qwen2.5-3B alone is published as 3.09B parameters, so it cannot be the language backbone of a 2B-total model. Either revise the total parameter target or select a smaller language backbone before freezing the architecture.
- Check the Qwen2.5-3B Research License before any commercial redistribution or deployment; its license is not the same as the Apache-2.0 license used by several other Qwen2.5 sizes.
