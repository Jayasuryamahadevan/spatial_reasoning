# LSF-VLM: Unified Edge Spatial Reasoning Architecture

**Status:** Architecture proposal v1.0 — research specification, not yet empirically validated  
**Backbone:** LiquidAI/LFM3-VL-2B  
**Target:** One locally deployable vision-language-spatial model with a shared visual pass and typed spatial outputs.

## 1. Design objective

LSF-VLM is one multimodal model, not a runtime chain of LFM3 + GroundingDINO + SAM + Depth Anything.

At inference, only the LSF-VLM checkpoint and its runtime are required. GroundingDINO, SAM, Depth Anything, and optional larger VLMs are offline teachers used to create supervision. Their weights must not be required by the deployed artifact.

One request should produce:
- Natural-language spatial reasoning.
- Object instances and normalized bounding boxes.
- Phrase-conditioned instance masks.
- Relative monocular depth and uncertainty.
- Explicit object-to-object spatial relations.
- Optional temporal object state for future video/streaming extensions.

"Single model" means one parameter bundle, one public inference call, one shared visual feature extraction, and one typed result. Internally it may contain several small neural modules and exported graph partitions.

## 2. Backbone facts and constraints

The LFM3-VL-2B model card reports 1.92B parameters: 1.73B language model, 164M vision encoder, and 25M projector. Its language backbone has 24 layers: 13 gated short-convolution layers, 5 sliding-window attention layers, and 6 global-attention layers. Its vision encoder has 14 layers: 8 gated depthwise 2D-convolution layers and 6 bidirectional-attention layers. Image patches are 16x16 and each 2x2 patch block is merged into one language token.

The model is intended for on-device visual understanding, grounding, and embodied reasoning. Its model card cautions that it is not recommended as a general-purpose coding, multi-turn conversation, or complex-reasoning model. We must benchmark our target tasks rather than infer capability from the category label.

### Parameter budget

The base already consumes approximately 1.92B parameters. A strict 2.00B ceiling leaves approximately 80M for all spatial additions and any newly introduced parameters that remain in the merged inference model.

Initial engineering budget (targets, not measured counts):

| Component | Target additional parameters |
|---|---:|
| Spatial Perceiver / latent bridge | 8–14M |
| Multi-scale spatial adapter | 4–8M |
| Object, box, relation heads | 3–6M |
| Dense mask decoder | 8–14M |
| Relative-depth and uncertainty head | 2–4M |
| Query routing, projections, norms | 2–4M |
| Optional flow-refinement head | 4–8M |
| **Total spatial additions** | **31–58M** |
| **Target merged total** | **1.95–1.98B** |

These are budget allocations, not instantiated parameter measurements. Add a parameter-count test and enforce the 2B ceiling. Do not count only trainable parameters while excluding frozen inference weights.

If additions exceed budget, reduce decoder width/depth or remove the optional flow branch.

## 3. Core architecture: one shared spatial token bus

### 3.1 Shared visual encoding

For image I, call the LFM3-VL vision encoder once.

Expose:
- F_v: final spatial vision features before reduction to language-token form, shape [B, N_v, D_v], plus 2D grid metadata.
- F_detail: selected intermediate convolution feature maps from the same vision encoder, where accessible, to recover boundaries and spatial detail.
- V_lfm: the normal projected visual tokens expected by the original LFM3 language pathway.

Preserve the pretrained LFM3 image-to-language path. Do not replace it with a new SigLIP encoder or discard it in the first implementation.

Feature tap points and dimensions must be read from the installed checkpoint/config and verified with a one-image probe. Do not hard-code dimensions copied from another Liquid model.

### 3.2 Perceiver Spatial Latent Bridge

Map the variable-size visual feature grid to a fixed-size workspace:

Z = Perceiver(F_v + PE_2D)

- Z has shape [B, 64, D_s].
- D_s starts at 512.
- PE_2D is a normalized 2D coordinate embedding.
- 64 learned latent queries.
- Two lightweight cross-attention blocks, followed by latent self-attention and normalization.

These 64 tokens are the canonical scene representation used by every spatial head. They do not replace all high-resolution features: dense segmentation also needs a small multi-scale detail path from F_detail.

### 3.3 Language-conditioned task queries

The original LFM3 image tokens and question pass through LFM3 once.

Let H_q be final-layer hidden states corresponding to the user question or selected phrase span. Construct typed task queries:

Q_task = W_q * Pool(H_q) + E_task

E_task is a learned task-type embedding for:
- scene summary
- object detection
- phrase grounding
- instance segmentation
- depth
- relation extraction

For multi-object requests, use a fixed set of object slots. For phrase grounding, pool only the referring-phrase token span when offsets are available; otherwise use the whole-question representation.

For strict gradient isolation, spatial heads receive stop_gradient(H_q) in the first training stage. Spatial losses therefore cannot update the language model through the question-conditioning path.

### 3.4 Spatial token routing

Every spatial task consumes the same Z and task-conditioned query tokens.

- Object queries cross-attend to Z and predict entity slots.
- Phrase queries select/refine an entity slot.
- Mask queries combine the selected entity representation, Z, and F_detail.
- Depth queries decode a dense relative-depth field from Z and F_detail.
- Relation queries predict edges between entity slots.

An entity has one stable entity_id. Its box, class, mask, depth summary, and relations all refer to that same ID. This prevents independent heads from returning incompatible object orderings.

## 4. Dense prediction heads

### 4.1 Object and box head

Use a DETR-style set predictor with at most 32 object slots.

Each slot emits objectness, class/text-alignment score, normalized xyxy box coordinates, a 512D entity embedding, and optional uncertainty.

Train with Hungarian matching, focal/classification loss, and L1 + generalized-IoU box losses.

For open-vocabulary grounding, compare entity embeddings against a phrase embedding from LFM3 question hidden states. Do not rely on the LLM generating raw floating-point coordinates as the only box mechanism.

### 4.2 Instance mask head

Use a shared pixel decoder and entity-conditioned mask prediction:

M_i = sigmoid(P_pixel(F_detail, Z) dot W_entity(E_i))

This is the direct one-pass mask baseline. Preserve explicit object-slot-to-mask correspondence.

**Conditional flow matching is an optional refinement experiment, not the mandatory first implementation.** If used, refine low-resolution mask logits or signed-distance fields rather than repeatedly regenerating unconstrained full-resolution binary tensors.

For continuous signed-distance target y and Gaussian noise x_0:

x_t = (1 - t) * x_0 + t * y  
v_target = y - x_0  
L_flow = mean((v_theta(x_t, t, condition) - v_target)^2)

Evaluate 1, 2, 4, and 8 solver steps against the direct-mask decoder. Keep flow only if the mask-quality gain justifies measured latency and memory. More steps are not automatically better for edge inference.

### 4.3 Depth head

Predict normalized relative inverse depth plus uncertainty:
- D_rel: [B, 1, H_d, W_d]
- U_depth: [B, 1, H_d, W_d]

Depth Anything V2 pseudo-labels are relative depth, not calibrated metres. Preserve this distinction in the dataset and API.

Use scale-invariant log-depth loss, gradient/edge-aware smoothness, and confidence-weighted uncertainty loss. Do not claim metric depth without metric supervision and calibration.

### 4.4 Relation head

Predict a sparse scene graph over entity slots:

(subject_entity_id, predicate, object_entity_id, confidence)

Initial predicates:
- left_of / right_of
- above / below
- inside / contains
- overlaps
- in_front_of / behind, only where depth evidence supports it
- near / far, with calibrated uncertainty

Relations that cannot be inferred reliably from a monocular image should be marked uncertain, not guessed.

## 5. Token protocol and public output

Do not serialize dense masks or depth maps as thousands of autoregressive text tokens. Use typed metadata plus binary tensor payloads.

Public request fields:
- image
- question
- tasks: reason, boxes, masks, depth, relations

Public response fields:
- text
- entities: entity_id, label, normalized xyxy box, score, mask reference, mean relative depth
- relations: subject_id, predicate, object_id, confidence
- spatial_payload: mask encoding, depth encoding, depth shape, coordinate frame

Dense mask data can be returned as COCO RLE or a binary payload. Depth can be returned as float16 relative inverse depth. Coordinate frame must be defined against the original image.

Internal token contract:

| Token family | Role |
|---|---|
| Native LFM3 image tokens | Preserve pretrained vision-language understanding |
| 64 spatial latents | Shared scene/spatial workspace |
| Task query embedding | Select operation and condition the decoder |
| Object query slots | Stable entity identities |
| Phrase-span embedding | Refer to a named object in the question |
| Relation query | Produce entity-to-entity edges |

Task query embeddings may be learned soft tokens inside the model; they do not need literal strings in user-visible output. The runtime passes task types as typed IDs. This is more deterministic than waiting for generation to emit a SEG token, interrupting decoding, and resuming it.

## 6. One-call inference graph

    IMAGE + QUESTION + TASKS
                 |
       image preprocessing once
                 |
       LFM3 vision encoder once
             /             \
    native visual tokens    spatial features
            |                    |
      LFM3 projector       Perceiver bridge
            |                    |
     LFM3 language          64 latents
         prefill                  |
            |               shared spatial bus
   question hidden states          |
            |                +-----+-------+
   detached phrase/task Q       box  mask  depth/relation
            |                    |    |      |
            +--------------------+----+------+
                                 |
                             entity table
                                 |
                     text + boxes + masks
                     + depth + relations

Text decoding may continue autoregressively after prefill, but spatial heads run from the same encoded image and same prefill. The spatial API must not require generation of a special token followed by a pause, hidden-state extraction, and resumed decoding.

## 7. Training and gradient routing

First pass: freeze the LFM3 backbone. Train the spatial bridge and spatial heads only.

Loss:

L_total = L_text
        + lambda_box * L_box
        + lambda_mask * L_mask
        + lambda_depth * L_depth
        + lambda_rel * L_rel
        + lambda_flow * L_flow

Routing:
- Text loss updates permitted language-model LoRA parameters.
- Spatial losses update the Perceiver, spatial adapters, and heads through F_v.
- Spatial query context is detached from LFM3 hidden states in the first stage.
- Do not detach F_v if the vision encoder is intended to train. Do detach it when the first experiment freezes the entire backbone.
- Assert gradients on every parameter group in unit tests; do not infer correct routing merely because a detach appears in the code.

Suggested schedule:
1. Frozen backbone; train bridge + heads.
2. Add LoRA to selected LFM3 language layers using text/structured-output losses only.
3. Optionally add small vision adapters if spatial quality is inadequate.
4. Joint low-rate calibration only after gradient-route and regression tests pass.

Do not assume standard QLoRA targets work unchanged on LFM3. Its mixed convolution/attention architecture differs from an all-attention decoder. Inspect actual module names and validate PEFT injection before training.

## 8. Distillation data

The current 50K CC3M workflow is useful for captions and broad image coverage, but CC3M alone is not a spatial-supervision dataset.

Keep teacher responsibility explicit:
- GroundingDINO: phrase-conditioned boxes and confidence.
- SAM: instance masks prompted by teacher boxes.
- Depth Anything V2: relative inverse-depth pseudo-labels.
- LFM3-VL: optional semantic descriptions, relation proposals, and question-answer targets. Treat outputs as noisy labels and check independently.

Add appropriately licensed evaluation/training data for:
- referring-expression grounding;
- COCO/LVIS-style object and instance supervision;
- depth datasets with camera/scale metadata;
- indoor depth evaluation;
- spatial-relation and embodied benchmarks.

Deduplicate images and keep train/validation/test sources disjoint. Teacher labels must never be treated as ground truth for final validation.

## 9. Edge deployment design

### Logical artifact

Ship one versioned LSF model bundle containing:
- quantized LFM3-VL decoder weights;
- matching vision encoder/projector weights;
- spatial bridge and heads;
- tokenizer and image-processor configuration;
- task-query schema;
- runtime manifest and model/license notices.

The deployed bundle must not contain GroundingDINO, SAM, or Depth Anything teacher weights.

### Runtime choice

The official LFM3 model card lists ONNX exports with separate decoder and vision-encoder graphs, as well as GGUF for llama.cpp. ONNX Runtime is the preferred first integration target because the shared vision output and custom spatial heads need explicit tensor interfaces.

GGUF/llama.cpp is a useful local baseline, but do not assume it exposes all internal activations required by custom spatial heads. If activations are unavailable, use a custom llama.cpp hook/fork or keep the unified backend in ONNX Runtime.

### Memory and quantization

The official LFM3-VL ONNX distribution lists Q4 decoder/vision variants; GGUF also offers 4-bit variants. Compressed checkpoint size is not the complete runtime memory footprint.

Measure separately:
- model weight residency;
- image/vision activations;
- KV cache at a fixed context length;
- spatial feature and decoder buffers;
- runtime/workspace allocations;
- process RSS and GPU peak allocated/reserved memory.

For a 4GB RTX 3050, start with Q4 weights, short context, one image, conservative resolution, and partial CPU offload. A strict process RSS below 4GB is not assumed feasible; first define whether the limit means VRAM, model-file size, or total resident memory.

## 10. Required research ablations

1. LFM3-VL baseline versus LSF-VLM.
2. Native LFM visual tokens only versus native tokens + 64 spatial latents.
3. 16 / 32 / 64 / 128 latent tokens.
4. Direct masks versus flow-refined masks at 1 / 2 / 4 / 8 steps.
5. With and without high-resolution feature skips.
6. Detached language query versus controlled joint gradient.
7. Shared object slots versus independent per-task detections.
8. FP16 versus INT8/INT4 runtime quality and latency.

Report accuracy alongside latency, peak memory, parameter count, and power where hardware measurement is available.

## 11. Acceptance gates

### Architecture probe
- Load LFM3-VL using the currently documented Transformers API.
- Extract native visual tokens, vision features, and language hidden states.
- Verify one image goes through the vision encoder exactly once.
- Record tensor shapes, dtypes, device placement, total parameter count, latency, and peak memory.
- Verify whether pre-merge spatial features or intermediate convolution maps are exposed.

### Spatial module probe
- 64 latent tokens have the expected fixed shape.
- Boxes are finite, normalized, and ordered.
- Every entity ID maps to its corresponding mask.
- Depth is finite and explicitly tagged as relative.
- Relation IDs reference existing entities.
- Spatial losses have zero gradients on detached/frozen language parameters and nonzero gradients on intended spatial parameters.

### Edge probe
- One local request returns text + boxes + masks + depth + relations.
- No teacher model is loaded.
- One vision encoding per image request is verified with an instrumented counter.
- Record P50/P95 latency and memory at fixed image size, context length, and hardware.

Do not accept benchmark claims until measured on held-out data with documented splits and reproducible evaluation scripts.

## 12. Immediate next implementation step

Do not start the 50K annotation run yet.

Build experiments/lfm3_probe.py first. It must:
1. Load LiquidAI/LFM3-VL-2B using the currently documented Transformers API (model card specifies transformers >=5.21.0 and trust_remote_code=True).
2. Run one 384/448-resolution image with a short question.
3. Capture vision encoder outputs and language hidden states without changing the checkpoint.
4. Print shapes, dtypes, devices, parameter counts, inference time, peak allocated/reserved VRAM.
5. Verify whether pre-merge spatial features or intermediate convolution maps are accessible for the dense decoder.

Only after the probe passes should we implement the Perceiver bridge. Keep LFM3 runtime dependencies separate from the existing teacher-pipeline environment until Transformers compatibility is verified.
