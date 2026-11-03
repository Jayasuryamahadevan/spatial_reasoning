from __future__ import annotations

import os
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
        detach_spatial_language_context: bool = True,
        detach_spatial_prefix_input: bool = True,
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
        # Start as a neutral prefix. Stage-2 language adaptation learns its scale/content.
        nn.init.zeros_(self.spatial_to_language[-1].weight)
        nn.init.zeros_(self.spatial_to_language[-1].bias)
        self.layer_fusion_logits = nn.Parameter(torch.zeros(4))
        self.freeze_backbone = freeze_backbone
        self.detach_spatial_language_context = detach_spatial_language_context
        self.detach_spatial_prefix_input = detach_spatial_prefix_input
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
        # Keep compact spatial heads beside the vision tower.
        vision_parameter = next(self.backbone.model.vision_tower.parameters())
        spatial_device = vision_parameter.device
        spatial_dtype = vision_parameter.dtype
        self.spatial.to(device=spatial_device, dtype=spatial_dtype)
        self.spatial_to_language.to(device=spatial_device, dtype=spatial_dtype)
        self.spatial_type_embedding.data = self.spatial_type_embedding.data.to(
            device=spatial_device, dtype=spatial_dtype
        )
        self.layer_fusion_logits.data = self.layer_fusion_logits.data.to(
            device=spatial_device, dtype=torch.float32
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "LiquidAI/LFM3-VL-2B",
        spatial_config: SpatialCoreConfig | None = None,
        torch_dtype: torch.dtype | None = torch.float16,
        device_map: str | dict[str, int] | None = None,
        freeze_backbone: bool = True,
        **kwargs: Any,
    ) -> "UnifiedLSFVLM":
        from transformers import AutoModelForImageTextToText

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": torch_dtype,
            **kwargs,
        }
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        elif torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            load_kwargs["device_map"] = {"": local_rank}
        backbone = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
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
        weights = torch.softmax(self.layer_fusion_logits[-count:], dim=0).to(
            device=candidate_layers[-1].device, dtype=candidate_layers[-1].dtype
        )
        fused = sum(weight * state for weight, state in zip(weights, candidate_layers))
        channels = fused.shape[-1]
        return fused.reshape(1, height, width, channels).permute(0, 3, 1, 2).contiguous()

    def _pool_language_context(
        self,
        language_hidden: Tensor,
        question_mask: Tensor | None,
        insertion_index: int,
        prefix_inserted: bool,
    ) -> Tensor:
        if question_mask is None:
            # Fallback: pool all attended language tokens after the image block.
            mask = torch.ones(
                language_hidden.shape[:2],
                device=language_hidden.device,
                dtype=language_hidden.dtype,
            )
            mask[:, :insertion_index] = 0
            if prefix_inserted:
                mask[
                    :,
                    insertion_index : insertion_index + self.config.num_spatial_latents,
                ] = 0
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


    def _pool_phrase_contexts(
        self,
        language_hidden: Tensor,
        phrase_token_masks: Tensor | None,
        insertion_index: int,
        prefix_inserted: bool,
    ) -> Tensor | None:
        if phrase_token_masks is None:
            return None
        if phrase_token_masks.ndim != 3 or phrase_token_masks.shape[0] != 1:
            raise ValueError("phrase_token_masks must be [1,num_phrases,original_sequence_length]")
        if phrase_token_masks.shape[-1] == language_hidden.shape[1]:
            masks = phrase_token_masks.to(language_hidden.device, language_hidden.dtype)
        elif not prefix_inserted:
            masks = phrase_token_masks.to(language_hidden.device, language_hidden.dtype)
        else:
            zeros = torch.zeros(
                (
                    phrase_token_masks.shape[0],
                    phrase_token_masks.shape[1],
                    self.config.num_spatial_latents,
                ),
                device=language_hidden.device,
                dtype=language_hidden.dtype,
            )
            masks = torch.cat(
                (
                    phrase_token_masks[:, :, :insertion_index],
                    zeros,
                    phrase_token_masks[:, :, insertion_index:],
                ),
                dim=-1,
            ).to(language_hidden.device, language_hidden.dtype)
        weights = masks / masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.einsum("bpl,bld->bpd", weights, language_hidden)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor,
        image_grid_thw: Tensor,
        rgb: Tensor,
        question_mask: Tensor | None = None,
        phrase_token_masks: Tensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool = False,
        language_path_grad: bool | None = None,
        insert_spatial_prefix: bool = True,
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

        # 2) Produce the canonical spatial memory and the projected full patch
        #    context. The 64 latents are global scene context; entity queries
        #    also attend to the full patch memory for precise localization.
        spatial_tokens, visual_memory = self.spatial.encode_scene(
            vision_map, return_memory=True
        )
        prefix_source = spatial_tokens.detach() if self.detach_spatial_prefix_input else spatial_tokens
        spatial_prefix = self.spatial_to_language(prefix_source) + self.spatial_type_embedding

        # 3) Insert spatial latents into the causal LFM3 sequence immediately after
        #    the image block. These are continuous embeddings, not generated text.
        embedding_layer = self.backbone.model.get_input_embeddings()
        input_embeds = embedding_layer(input_ids)
        vision_end_id = int(self.backbone.config.vision_end_token_id)
        insertion_index = find_vision_end_insertion_index(input_ids, vision_end_id)
        if insert_spatial_prefix:
            new_ids, new_embeds, new_attention = insert_soft_spatial_tokens(
                input_ids=input_ids,
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                spatial_prefix=spatial_prefix,
                insertion_index=insertion_index,
                placeholder_token_id=int(self.backbone.config.pad_token_id),
            )
        else:
            new_ids = input_ids
            new_embeds = input_embeds
            new_attention = attention_mask

        new_labels = None
        if labels is not None:
            if insert_spatial_prefix:
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
            else:
                new_labels = labels

        # 4) Reuse the exact precomputed vision output. Passing mm_encoder_outputs
        #    prevents the LFM3 model from executing the vision tower a second time.
        if language_path_grad is None:
            language_path_grad = not self.freeze_backbone
        with torch.set_grad_enabled(torch.is_grad_enabled() and language_path_grad):
            language_outputs = self.backbone.model(
                input_ids=None,
                inputs_embeds=new_embeds,
                attention_mask=new_attention,
                mm_encoder_outputs={"image": vision_outputs},
                output_hidden_states=False,
                use_cache=use_cache,
                return_dict=True,
            )
        language_hidden = language_outputs.last_hidden_state
        logits = self.backbone.lm_head(language_hidden)
        rgb = rgb.to(device=vision_map.device, dtype=vision_map.dtype)

        # 5) Condition entity queries on the same LFM3 contextualized question.
        language_context = self._pool_language_context(
            language_hidden, question_mask, insertion_index, insert_spatial_prefix
        )
        phrase_contexts = self._pool_phrase_contexts(
            language_hidden, phrase_token_masks, insertion_index, insert_spatial_prefix
        )
        if self.detach_spatial_language_context:
            language_context = language_context.detach()
            if phrase_contexts is not None:
                phrase_contexts = phrase_contexts.detach()
        spatial_outputs = self.spatial(
            vision_map=vision_map,
            rgb=rgb,
            language_context=language_context,
            spatial_tokens=spatial_tokens,
            visual_memory=visual_memory,
            phrase_contexts=phrase_contexts,
        )

        result: dict[str, Any] = {
            **spatial_outputs,
            "language_logits": logits,
            "past_key_values": language_outputs.past_key_values,
            "expanded_input_ids": new_ids,
            "expanded_attention_mask": new_attention,
            "labels": new_labels,
        }
        return result

    @torch.inference_mode()
    def generate_answer(
        self,
        tokenizer: Any,
        max_new_tokens: int = 96,
        eos_token_id: int | list[int] | None = None,
        **forward_kwargs: Any,
    ) -> dict[str, Any]:
        """Run one unified prefill, return spatial heads, then decode the text answer.

        The image/vision encoder is evaluated once in forward. Subsequent text
        tokens are decoded from the returned LFM3 KV/recurrent cache and do not
        invoke the vision tower or any external teacher model.
        """
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        output = self.forward(**forward_kwargs, use_cache=True)
        cache = output["past_key_values"]
        attention_mask = output["expanded_attention_mask"]
        next_logits = output["language_logits"][:, -1, :]

        if eos_token_id is None:
            eos_token_id = getattr(self.backbone.generation_config, "eos_token_id", None)
        if eos_token_id is None:
            eos_ids: set[int] = set()
        elif isinstance(eos_token_id, int):
            eos_ids = {eos_token_id}
        else:
            eos_ids = {int(token_id) for token_id in eos_token_id}

        generated: list[Tensor] = []
        for _ in range(max_new_tokens):
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            generated.append(next_token)
            if eos_ids and int(next_token[0, 0].item()) in eos_ids:
                break
            if len(generated) >= max_new_tokens:
                break

            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ),
                dim=1,
            )
            decoded = self.backbone.model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = decoded.past_key_values
            next_logits = self.backbone.lm_head(decoded.last_hidden_state[:, -1, :])

        generated_ids = torch.cat(generated, dim=1)
        output["past_key_values"] = cache
        output["generated_token_ids"] = generated_ids
        output["answer_text"] = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return output

    def parameter_report(self) -> dict[str, int]:
        backbone_total = sum(parameter.numel() for parameter in self.backbone.parameters())
        spatial_total = sum(parameter.numel() for parameter in self.spatial.parameters())
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
            "spatial_addon_parameters": total - backbone_total,
            "model_parameters_total": total,
            "model_parameters_trainable": trainable,
        }
