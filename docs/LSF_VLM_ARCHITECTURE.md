# LSF-VLM Architecture

The authoritative source-grounded design, literature review, tensor contracts, training plan, edge deployment decision, and validation gates are maintained in:

- [Research review and architecture decision record](RESEARCH_REVIEW.md)

The model is one unified LFM3-VL-based checkpoint. GroundingDINO, SAM, and Depth Anything are permitted only as offline teacher models for distillation; they are not part of deployed inference.

Implementation entry points:
- \`lsf_vlm/modeling/lfm3_adapter.py\`
- \`lsf_vlm/modeling/spatial_core.py\`
- \`lsf_vlm/training/\`

Run the one-image integration gate before large-scale annotation or training:

\`\`\`bash
pip install -r requirements-lfm3.txt
python experiments/lsf_one_image_smoke.py --image /path/to/image.jpg
pytest -q tests/test_spatial_core.py
\`\`\`

This design is a research implementation under validation, not a claim that benchmark or edge targets have already been achieved.
