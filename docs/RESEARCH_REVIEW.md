# LSF-VLM Research Review and Architecture Decision Record

**Revision:** 2026-11-03  
**Decision:** Build one end-to-end LFM3-VL-based spatial reasoning model.  
**Research status:** Architecture grounded in source inspection; runtime and quality still require experimental validation.

## Executive decision

LSF-VLM must be one checkpoint and one end-to-end forward path that unifies:

1. Vision-language reasoning.
2. Object/phrase grounding and pixel-level segmentation.
3. Monocular relative-depth estimation.

The final runtime must not load GroundingDINO, SAM, or Depth Anything. Those systems may be used offline to generate weak supervision, but the deployed LSF-VLM must predict all outputs from its own weights.

The selected starting point is LiquidAI/LFM3-VL-2B. Its model card reports 1.92B parameters and native image/video input. Its text-only language response is not a dense segmentation/depth interface, so LSF adds a compact shared spatial representation and task heads.

The model is not yet proven to meet the requested metrics. In particular, the LFM3 card explicitly cautions against assuming strong general complex reasoning. We must compare the untouched backbone and LSF-VLM on held-out spatial benchmarks.

## 1. Source-level analysis of LFM3-VL

Primary sources:
- Model card: https://huggingface.co/LiquidAI/LFM3-VL-2B
- Configuration: https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/main/config.json
- Implementation: https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/main/modeling_lfm3_vl.py
- ONNX export: https://huggingface.co/LiquidAI/LFM3-VL-2B-ONNX

Reported configuration:

| Component | Value |
|---|---:|
| Total pretrained parameters | 1.92B |
| Language model | 1.73B |
| Vision encoder | 164M |
| Multimodal projector | 25M |
| Language hidden size | 2,048 |
| Vision hidden size | 1,024 |
| Projector hidden size | 4,096 |
| Language layers | 24 |
| Vision layers | 14 |
| Vision patch size | 16 |
| Spatial merge size | 2 |
| Language vocabulary | 128,000 |

The language backbone is hybrid: 13 gated short-convolution layers, 5 sliding-window-attention layers, and 6 global-attention layers. The vision tower alternates gated depthwise 2D convolutions and bidirectional attention.

### Important implementation finding

The official source defines Lfm3VlModel with:
- vision_tower
- multi_modal_projector
- language_model

Its get_image_features(pixel_values, image_grid_thw, ...) method:
1. Executes vision_tower once.
2. Retains its unmerged last_hidden_state in the returned model output.
3. Passes that state through the multimodal projector.
4. Stores the per-image projected tokens in pooler_output.

The vision model uses an output-capture decorator and declares hidden_states as recordable outputs. Thus output_hidden_states=True can expose intermediate vision-layer tensors. Those tensors share the same patch-grid resolution; they are not a hierarchical FPN.

The outer LFM3 model forward accepts mm_encoder_outputs. When supplied, it uses the precomputed image output and skips get_image_features. It scatters pooler_output into the image placeholders and passes the resulting embeddings to the LFM3 language model.

This gives us a source-grounded one-vision-pass integration:

1. Call get_image_features(..., output_hidden_states=True) once.
2. Use the unmerged [patch_count, 1024] features for the spatial path.
3. Use that exact output object as mm_encoder_outputs={"image": vision_outputs} for the language path.
4. Do not pass pixel_values again to that language forward.

This is stronger than an external second vision encoder. It lets the LSF spatial decoder use the original patch grid while preserving the pretrained image-token pathway.

For an initial still-image implementation, batch size is deliberately one. The adapter validates image_grid_thw, requires one temporal slice, and reshapes the patch sequence using the supplied (T,H,W) grid. Dynamic multi-image batching and video are later engineering tasks.

## 2. Research precedents and what LSF adopts

### X-Decoder — CVPR 2023

X-Decoder shows that pixel-level segmentation and language outputs can share a visual-semantic space. It uses generic and semantic queries to support diverse segmentation and vision-language tasks.

LSF adopts the shared semantic/pixel-space principle. It differs by using LFM3's pretrained image-language pathway and adds explicit depth prediction and stable entity slots.

Source: https://openaccess.thecvf.com/content/CVPR2023/html/Zou_Generalized_Decoding_for_Pixel_Image_and_Language_CVPR_2023_paper.html

### SEEM — NeurIPS 2023

SEEM shows how points, boxes, scribbles, masks, and text prompts can share a promptable segmentation interface. Its visual and semantic prompts are composed in a common decoder.

LSF adopts typed task/phrase conditioning, but avoids making autoregressive special-token generation the control plane for dense tasks. The runtime supplies typed phrase spans and task IDs directly.

Sources:
- https://papers.neurips.cc/paper_files/paper/2023/hash/3ef61f7e4afacf9a2c5b71c726172b86-Abstract-Conference.html
- https://www.microsoft.com/en-us/research/publication/segment-everything-everywhere-all-at-once/

### Mask2Former — CVPR 2022

Mask2Former uses query-based masks and masked attention to unify semantic, instance, and panoptic segmentation. The important design lesson is to predict a set of entities and associate each query with its own mask.

LSF adopts a compact entity-query decoder and query-conditioned masks rather than one independent segmentation network per task.

Source: https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html

### UniVLG — ICML 2025

UniVLG unifies 2D and 3D vision-language understanding through language-conditioned object queries and a mask decoder shared across modalities. It demonstrates that entity/query representations are a useful interface between language grounding and segmentation.

LSF adopts stable entity slots, language-conditioned query features, and entity-indexed masks. LSF's present target is monocular 2D RGB with relative depth, not full 3D reconstruction; camera-calibrated 3D lifting is a separate future extension.

Sources:
- https://proceedings.mlr.press/v267/jain25b.html
- https://univlg.github.io/

### EMUFormer — IJCV 2026

EMUFormer studies joint semantic segmentation and monocular depth with a shared encoder, parallel decoders, and uncertainty distillation. It uses Gaussian negative log-likelihood for depth and distills teacher uncertainty.

LSF adopts shared features, depth uncertainty, and uncertainty-aware distillation. Its language-conditioned entity slots and dense mask outputs make the overall task broader than semantic segmentation plus depth.

Source: https://link.springer.com/article/10.1007/s11263-026-02751-0

### MulT — CVPR 2022

MulT is a direct precedent for joint dense prediction. It shares an image representation and uses cross-task attention for depth, semantic segmentation, reshading, surface normals, keypoints, and edges. This means "one encoder plus depth and segmentation heads" is not a novelty claim.

LSF should go beyond parallel outputs: the same learned entity slots must bind language-grounded objects to their masks, boxes, depth summaries, and relation-graph nodes. The dense depth field and instance masks also share a pixel representation.

Source: https://openaccess.thecvf.com/content/CVPR2022/html/Bhattacharjee_MulT_An_End-to-End_Multitask_Learning_Transformer_CVPR_2022_paper.html

### UFO — NeurIPS 2025

UFO is an especially close precedent. It unifies object-level detection, pixel-level segmentation, and image-level vision-language tasks through an open-ended language interface and an embedding-retrieval approach for masks. Its reported results demonstrate that an MLLM can be trained as one fine-grained perception model rather than a tool orchestrator.

This rules out claiming that LSF is the first unified VLM for detection/segmentation. Our differentiated hypothesis must include dense monocular relative depth, a shared entity-indexed spatial graph, continuous spatial memory integrated into LFM3's causal sequence, and a deployment budget tested on constrained edge hardware.

Source: https://proceedings.neurips.cc/paper_files/paper/2025/hash/78a241e1d2ff2366af8cc9a459875a01-Abstract-Conference.html

### Research novelty position

A shared encoder, Perceiver latents, mask queries, task-conditioned prediction, and joint segmentation/depth are not individually novel. We must not claim them as standalone inventions.

The research contribution to test is the combination of:
- LFM3's hybrid native vision-language backbone;
- one vision-tower execution whose unmerged patch memory is shared by the LFM3 language path, Perceiver, and dense heads;
- continuous spatial latents inserted into the same causal language sequence;
- one entity table shared by phrase grounding, boxes, masks, depth summaries, and relations;
- one shared pixel feature tensor used for both masks and depth;
- explicit gradient routing and edge-budgeted distillation.

This is a research hypothesis, not a novelty guarantee. A systematic related-work search and direct ablations against UFO, X-Decoder, MulT, Mask2Former, and language-reasoning segmentation baselines are required before a publication claim.

## 3. Chosen LSF-VLM architecture

### 3.1 One image pass

Inputs:
- processed LFM3 patch tensor;
- image_grid_thw;
- original RGB tensor for the integrated low-cost detail stem;
- tokenized text question and optional phrase-span masks.

The LFM3 vision tower runs once and returns unmerged patch features plus its projected image tokens.

Let:
- F_v be the last/fused unmerged vision feature map, [B,1024,H_p,W_p];
- V_lm be LFM3's normal merged image tokens, [N_img,2048].

### 3.2 Shared Perceiver spatial memory

The current implementation uses:
- 64 learned latent queries;
- spatial width 256;
- 2 Perceiver blocks;
- learned 2D coordinate embeddings;
- cross-attention to the unmerged LFM3 patch grid.

Z = Perceiver(F_v + PE_2D), shape [B,64,256].

This is the single spatial token bus. It is not used as the only input to pixel prediction; coarse latent compression alone would lose boundary detail.

### 3.3 Spatial soft-prefix token passing

Project the 64 spatial tokens from width 256 to LFM3 language width 2,048.

Insert them as continuous input embeddings immediately after the image's vision_end marker and before the question tokens:

[system][image placeholders][vision_end][64 spatial soft tokens][question][assistant]

Placeholder token IDs are bookkeeping only. inputs_embeds at those positions are replaced by the learned spatial vectors. The inserted positions receive attention-mask value 1.

The same precomputed vision output object is supplied to LFM3 via mm_encoder_outputs, so the language path consumes its normal projected visual tokens without repeating the vision pass.

The question tokens can therefore attend causally to:
- native LFM3 image tokens;
- the inserted spatial-token memory;
- preceding text tokens.

This is the central language/spatial integration. It is not a separate LLM call after an independent vision model.

### 3.4 Contextual entity queries

The LFM3 prefill returns contextualized question states. The model pools the question span, projects it to width 256, and adds it to 32 learned entity queries.

The entity decoder applies self-attention between entity queries and cross-attention to the concatenation of Z and the full projected patch memory [B,N_patch,256]. This avoids asking 64 compressed scene tokens alone to localize small objects. The 64 latents carry global scene context; the full patch memory retains spatial detail for entity localization.

Every entity slot produces:
- entity embedding;
- objectness;
- normalized xyxy box;
- phrase-alignment scores when phrase-span embeddings are provided.

The same entity index is used for its mask and graph relations. This avoids disconnected heads returning different object orderings.

### 3.5 Shared dense pixel feature path

The LFM3 vision tower has a flat patch grid at 1/16 scale; its intermediate layers are not a four-stage feature pyramid. We therefore do not claim that LFM3 itself provides high-resolution FPN features.

The current integrated model adds a small RGB detail stem inside the same checkpoint. LFM3's intermediate feature taps share one patch-grid resolution rather than forming an FPN, so the detail stem restores local image boundaries while the final/unmerged LFM3 map supplies semantic features.
- two stride-2 convolution stages to reach 1/4 image resolution;
- a projection of the unmerged LFM3 patch map;
- bilinear fusion at 1/4 scale.

This is not an external pretrained visual model. It is a lightweight learned branch that preserves local edges and texture. The dense head is expected to fuse semantic LFM3 features with local image details.

The shared pixel tensor feeds both:
- entity-conditioned mask logits via a dot product between entity mask embeddings and normalized pixel features;
- relative inverse-depth and log-variance prediction.

Masks are emitted at 1/4 scale and resized to original dimensions by the inference API. Depth is emitted at original input resolution.

### 3.6 Relation graph

A pairwise relation head consumes (entity_i, entity_j, box_delta, center_delta, size_ratio) and predicts a fixed predicate vocabulary, including a no-relation class.

Initial predicates:
- no_relation
- left_of / right_of
- above / below
- inside / contains
- overlaps
- in_front_of / behind

Depth-based predicates must be uncertainty-aware. Monocular relative depth does not by itself provide metric distance or calibrated 3D geometry.

## 4. Token and tensor contract

| Tensor | Shape | Use |
|---|---|---|
| F_v | [B,1024,H_p,W_p] | Unmerged LFM3 spatial features |
| V_lm | [N_img,2048] | Native image tokens for LFM3 |
| Z | [B,64,256] | Shared spatial token bus |
| Z_lm | [B,64,2048] | Continuous soft prefix for language model |
| H_q | [B,2048] | Pooled contextual question representation |
| E | [B,32,256] | Shared entity slots |
| P | [B,128,H/4,W/4] | Shared dense pixel features |
| B_box | [B,32,4] | Normalized xyxy boxes |
| M | [B,32,H/4,W/4] | Per-entity mask logits |
| D | [B,1,H,W] | Positive relative inverse-depth output |
| U | [B,1,H,W] | Depth log variance |
| R | [B,32,32,10] | Pairwise relation logits (including no_relation) |

Text generation, dense masks, and depth maps must not be serialized as a long sequence of generated numeric tokens. Return structured JSON metadata plus binary/RLE tensor payloads.

## 5. Gradient routing

The current adapter exposes two explicit switches:

- detach_spatial_prefix_input=True: language loss can train the spatial-to-language projection, but it cannot update the Perceiver through the inserted prefix.
- detach_spatial_language_context=True: spatial losses cannot propagate through the pooled LFM3 question hidden state into the language model.

Spatial losses still propagate through the spatial core and, when unfrozen, the LFM3 vision features. This is the desired initial routing.

Initial stages:
1. Freeze LFM3; train Perceiver, entity queries, pixel decoder, depth and relation heads.
2. Add selected language-model LoRA only after inspecting the hybrid module names. Text loss trains LoRA. Spatial query context stays detached.
3. Optionally unfreeze small vision adapters if dense-task performance requires it.
4. Test joint gradient routing with parameter-group gradient assertions before broad training.

Do not apply a generic all-attention QLoRA target list without inspecting LFM3's convolution and attention module structure.

## 6. Data and distillation implications

CC3M is an image-URL/caption collection, not a dense spatial dataset. Its official distribution reports 3,318,333 training image-caption pairs, with captions averaging 10.3 tokens. It has no human annotators for the caption collection.

CC3M is useful for broad image/text alignment and pseudo-label collection, but it is insufficient alone for:
- reliable instance masks;
- calibrated depth;
- pairwise spatial relations;
- grounded phrase-span evaluation.

Teacher usage:
- GroundingDINO: phrase-conditioned boxes.
- SAM: boxes-to-mask pseudo labels.
- Depth Anything V2: relative depth pseudo labels.
- LFM3-VL: candidate captions, relation proposals and structured answers; always quality-filter.

Add separately licensed spatial datasets for instance masks, referring expressions, indoor/outdoor depth, and spatial relation reasoning. Keep validation/test splits disjoint from pseudo-label training. Never report validation quality against a teacher's own generated labels as if those were ground truth.

## 7. Training objectives

Initial total objective:

L = λ_txt L_txt + λ_obj L_obj + λ_box L_box + λ_mask (L_focal + L_dice) + λ_depth L_depth_uncertainty + λ_phrase L_phrase + λ_rel L_relation

Recommended components:
- language: causal cross-entropy with -100 labels at inserted spatial-prefix positions;
- objectness: BCE/focal over matched and unmatched entity slots;
- boxes: Hungarian assignment, L1 and generalized-IoU;
- masks: per-entity focal + Dice;
- depth: scale-invariant log inverse-depth residual with learned log variance and edge-aware smoothness;
- phrase grounding: cross-entropy over entity slots for each supplied phrase;
- relations: cross-entropy over pairwise predicates, ignoring invalid/unannotated edges.

Conditional flow matching is not in the initial runtime model. It is an optional ablation after direct masks work. Flow should refine a low-resolution continuous mask/logit or signed-distance representation, and it must beat direct mask prediction on quality/latency before inclusion.

## 8. Model-size and edge constraints

The LFM3 base already has about 1.92B parameters. The current spatial core is deliberately compact, but the adapter also introduces spatial-to-language projection and type embeddings; the final parameter report must include every module. Do not claim a final parameter total until UnifiedLSFVLM.parameter_report() is executed on the actual loaded checkpoint. LFM3-VL is listed under the LFM Open License v1.0; review that license and every dataset/teacher license before redistributing weights or building a commercial artifact.

The official LFM3-VL ONNX card reports a Q4 decoder + Q8 vision package around 1.69 GB and a Q4 decoder + FP16 vision GPU package around 1.69 GB. These are artifact sizes, not total runtime RAM/VRAM.

For the RTX 3050 4GB target:
- start at Q4 language decoder;
- keep the vision path FP16/INT8 depending accuracy;
- use one image per request and short context;
- offload a measured portion of the language decoder to system RAM;
- keep spatial heads FP16 initially;
- profile GPU memory, process RSS and latency separately.

A <4GB total memory claim is not supported. The target must be specified as either VRAM, model artifact size, or combined process RSS.

## 9. Experiments and acceptance criteria

### Backbone/interface
- Verify get_image_features(..., output_hidden_states=True) returns final unmerged features and intermediate vision outputs.
- Instrument the vision tower and assert exactly one call per image.
- Verify mm_encoder_outputs reuse produces the same LFM3 image-token path as the normal forward.
- Verify 64 soft tokens are inserted after vision_end and attended by question tokens.
- Run batch size 1 first.

### Spatial core
- Run pytest tests/test_spatial_core.py.
- Record exact spatial-core parameter count.
- Verify all output shapes and finite values.
- Verify boxes are ordered and normalized.
- Verify mask entity IDs match entity slots.
- Verify depth output positive and its semantics explicitly tagged as relative inverse depth.
- Verify relation IDs refer to valid entity slots.

### Gradient routing
- Backpropagate a spatial-only loss and assert no LFM3 language-model gradients through question conditioning.
- Assert nonzero gradients on Perceiver, entity decoder and task heads.
- If vision adapters are unfrozen, assert allowed vision-layer gradients.
- Backpropagate text loss separately and verify the intended text/LoRA and spatial-prefix projection gradients.

### Quality
Compare the untouched LFM3 baseline to LSF-VLM on held-out:
- referring-expression grounding;
- instance segmentation;
- relative-depth metrics;
- spatial relation accuracy;
- question answering with spatial relations.

Report box AP/IoU, mask mIoU/AP, depth AbsRel and δ thresholds, relation accuracy/F1, language accuracy, calibration, P50/P95 latency, peak memory, and model parameters.

No proposed target metric is claimed achieved until the corresponding experiment is measured.

## 10. Implementation status and next gates

Implemented in this repository:
- lsf_vlm/modeling/spatial_core.py
- lsf_vlm/modeling/lfm3_adapter.py (single vision-tower execution; now passes the full projected patch memory to entity queries)
- lsf_vlm/training/losses.py
- tests/test_spatial_core.py

Not yet validated:
- loading the actual LFM3 checkpoint through the wrapper;
- feature-grid reshaping on real processor output;
- one-pass vision instrumentation;
- unit-test execution in Kaggle;
- training step on T4;
- accuracy, latency and memory results.

The next gate is the one-image LFM3 integration test, followed by the CPU shape/gradient tests. Do not generate 50K annotations until these gates pass.
