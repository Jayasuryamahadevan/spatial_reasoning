from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .spatial_core import (
    LSFSpatialCore,
    SpatialCoreConfig,
    find_vision_end_insertion_index,
    insert_soft_spatial_tokens,
)


class UnifiedLSFVLM(nn.Module):
    """LFM3-VL plus one shared spatial-token workspace and spatial task heads.

    The vision tower is executed exactly once. Its unmerged patch features feed
    LSFSpatialCore; the same returned multimodal encoder output is then supplied
    to the original LFM3 language path through mm_encoder_outputs.

    This integration intentionally starts with batch_size=1 and one still image.
    """

    def __init__(
        self,
        lfm3_backbone: nn.Module,
        spatial_config: SpatialCoreConfig | None = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = lfm3_backbone
        self.config = spatial_config or SpatialCoreConfig(
            vision_dim=int(self.backbone.config.vision_config.hidden_size),
            language_dim=int(self.backbone.config.text_config.hidden_size),
        )
        self.spatial = LSFSpatialCore(self.config)
        self.spatial_to_language = nn.Sequential(
            nn.LayerNorm(self.config.spatial_dim),
            nn.Linear(self.config.spatial_dim, self.config.language_dim),
        )
        self.spatial_type_embedding = nn.Parameter(
            torch.zeros(1, self.config.num_spatial_latents, self.config.language_dim)
        )
        self.layer_fusion_logits = nn.Parameter(torch.zeros(4))
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "LiquidAI/LFM3-VL-2B",
        spatial_config: SpatialCoreConfig | None = None,
        torch_dtype: torch.dtype | None = torch.bfloat16,
        device_map: str | dict[str, int] | None = "auto",
        freeze_backbone: bool = True,
        **kwargs: Any,
    ) -> "UnifiedLSFVLM":
        from transformers import AutoModelForImageTextToText

        backbone = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch_dtype,
            device_map=device_map,
            **kwargs,
        )
        return cls(backbone, spatial_config=spatial_config, freeze_backbone=freeze_backbone)

    def _vision_map(self, vision_outputs: Any, image_grid_thw: Tensor) -> Tensor:
        if image_grid_thw.ndim != 2 or image_grid_thw.shape[0] != 1:
            raise ValueError("Initial LSF integration supports exactly one image per forward call.")
        temporal, height, width = (int(value) for value in image_grid_thw[0].tolist())
        if temporal != 1:
            raise ValueError("Initial LSF integration supports still images, not video frames.")
        hidden_states = getattr(vision_outputs, "hidden_states", None)
        candidate_layers = list(hidden_states[-4:]) if hidden_states else []
        if not candidate_layers:
            candidate_layers = [vision_outputs.last_hidden_state]
        if candidate_layers[-1].shape[0] != height * width:
            raise RuntimeError(
                "Vision token count does not match image_grid_thw. "
                f"tokens={candidate_layers[-1].shape[0]}, grid={height}x{width}."
            )
        count = len(candidate_layers)
        weights = torch.softmax(self.layer_fusion_logits[-count:], dim=0)
        fused = sum(weight * state for weight, state in zip(weights, candidate_layers))
        channels = fused.shape[-1]
        return fused.reshape(1, height, width, channels).permute(0, 3, 1, 2).contiguous()

    def _pool_language_context(
        self,
        language_hidden: Tensor,
        question_mask: Tensor | None,
        insertion_index: int,
    ) -> Tensor:
        if question_mask is None:
            # Fallback: pool all attended language tokens after the image block.
            mask = torch.ones(
                language_hidden.shape[:2],
                device=language_hidden.device,
                dtype=language_hidden.dtype,
            )
            mask[:, :insertion_index] = 0
        else:
            if question_mask.ndim != 2 or question_mask.shape[0] != 1:
                raise ValueError("question_mask must be [1, original_sequence_length]")
            if question_mask.shape[1] == language_hidden.shape[1]:
                mask = question_mask.to(device=language_hidden.device, dtype=language_hidden.dtype)
            else:
                zeros = torch.zeros(
                    (1, self.config.num_spatial_latents),
                    device=language_hidden.device,
                    dtype=language_hidden.dtype,
                )
                mask = torch.cat(
                    (
                        question_mask[:, :insertion_index],
                        zeros,
                        question_mask[:, insertion_index:],
                    ),
                    dim=1,
                ).to(language_hidden.device, language_hidden.dtype)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (language_hidden * mask.unsqueeze(-1)).sum(dim=1) / denominator

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor,
        image_grid_thw: Tensor,
        rgb: Tensor,
        question_mask: Tensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool = False,
    ) -> dict[str, Any]:
        if input_ids.shape[0] != 1:
            raise ValueError("Initial LSF integration supports batch_size=1.")
        if pixel_values is None or image_grid_thw is None:
            raise ValueError("pixel_values and image_grid_thw are required.")
        if rgb.shape[0] != 1:
            raise ValueError("rgb batch size must be one.")

        # 1) Execute LFM3's vision tower once and retain its patch-grid features.
        vision_outputs = self.backbone.model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )
        vision_map = self._vision_map(vision_outputs, image_grid_thw)

        # 2) Produce the canonical spatial memory.
        spatial_tokens = self.spatial.encode_scene(vision_map)
        spatial_prefix = self.spatial_to_language(spatial_tokens) + self.spatial_type_embedding

        # 3) Insert spatial latents into the causal LFM3 sequence immediately after
        #    the image block. These are continuous embeddings, not generated text.
        embedding_layer = self.backbone.model.get_input_embeddings()
        input_embeds = embedding_layer(input_ids)
        vision_end_id = int(self.backbone.config.vision_end_token_id)
        insertion_index = find_vision_end_insertion_index(input_ids, vision_end_id)
        new_ids, new_embeds, new_attention = insert_soft_spatial_tokens(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            spatial_prefix=spatial_prefix,
            insertion_index=insertion_index,
            placeholder_token_id=int(self.backbone.config.pad_token_id),
        )

        new_labels = None
        if labels is not None:
            ignore = torch.full(
                (labels.shape[0], self.config.num_spatial_latents),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            new_labels = torch.cat(
                (labels[:, :insertion_index], ignore, labels[:, insertion_index:]),
                dim=1,
            )

        # 4) Reuse the exact precomputed vision output. Passing mm_encoder_outputs
        #    prevents the LFM3 model from executing the vision tower a second time.
        language_outputs = self.backbone.model(
            input_ids=new_ids,
            inputs_embeds=new_embeds,
            attention_mask=new_attention,
            mm_encoder_outputs={"image": vision_outputs},
            output_hidden_states=True,
            use_cache=use_cache,
            return_dict=True,
        )
        language_hidden = language_outputs.last_hidden_state
        logits = self.backbone.lm_head(language_hidden)

        # 5) Condition entity queries on the same LFM3 contextualized question.
        language_context = self._pool_language_context(
            language_hidden, question_mask, insertion_index
        )
        spatial_outputs = self.spatial(
            vision_map=vision_map,
            rgb=rgb,
            language_context=language_context,
            spatial_tokens=spatial_tokens,
        )

        result: dict[str, Any] = {
            **spatial_outputs,
            "language_logits": logits,
            "language_hidden_states": language_outputs.hidden_states,
            "past_key_values": language_outputs.past_key_values,
            "image_hidden_states": language_outputs.image_hidden_states,
            "expanded_input_ids": new_ids,
            "expanded_attention_mask": new_attention,
            "labels": new_labels,
        }
        return result

    def parameter_report(self) -> dict[str, int]:
        backbone_total = sum(parameter.numel() for parameter in self.backbone.parameters())
        spatial_total = sum(parameter.numel() for parameter in self.spatial.parameters())
        prefix_total = sum(
            parameter.numel()
            for module in (self.spatial_to_language, self)
            for parameter in module.parameters()
            if parameter is not None
        )
        # Deduplicate shared parameters by identity.
        seen = set()
        total = 0
        trainable = 0
        for parameter in self.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            total += parameter.numel()
            if parameter.requires_grad:
                trainable += parameter.numel()
        return {
            "backbone_parameters": backbone_total,
            "spatial_core_parameters": spatial_total,
            "model_parameters_total": total,
            "model_parameters_trainable": trainable,
        }
