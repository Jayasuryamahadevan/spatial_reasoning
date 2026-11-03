from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SpatialCoreConfig:
    vision_dim: int = 1024
    language_dim: int = 2048
    spatial_dim: int = 256
    num_spatial_latents: int = 64
    num_entities: int = 32
    num_heads: int = 8
    perceiver_layers: int = 2
    entity_layers: int = 2
    pixel_dim: int = 128
    num_relations: int = 9
    output_stride: int = 4

    def validate(self) -> None:
        if self.spatial_dim % self.num_heads:
            raise ValueError("spatial_dim must be divisible by num_heads")
        if self.num_spatial_latents < 1 or self.num_entities < 1:
            raise ValueError("Token and entity counts must be positive")
        if self.output_stride != 4:
            raise ValueError("The initial detail stem supports output_stride=4")


class PerceiverBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 3) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, queries: Tensor, context: Tensor) -> Tensor:
        q = self.query_norm(queries)
        kv = self.context_norm(context)
        update, _ = self.cross_attn(q, kv, kv, need_weights=False)
        queries = queries + update
        q = self.self_norm(queries)
        update, _ = self.self_attn(q, q, q, need_weights=False)
        queries = queries + update
        return queries + self.ffn(self.ffn_norm(queries))


class PerceiverSpatialBridge(nn.Module):
    """Compress a variable visual patch grid into a fixed spatial token memory."""

    def __init__(self, config: SpatialCoreConfig) -> None:
        super().__init__()
        dim = config.spatial_dim
        self.input_proj = nn.Linear(config.vision_dim, dim)
        self.position_proj = nn.Sequential(
            nn.Linear(2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.latent_queries = nn.Parameter(torch.randn(1, config.num_spatial_latents, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [PerceiverBlock(dim, config.num_heads) for _ in range(config.perceiver_layers)]
        )
        self.output_norm = nn.LayerNorm(dim)

    @staticmethod
    def _coordinates(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2)

    def forward(self, vision_map: Tensor) -> Tensor:
        if vision_map.ndim != 4:
            raise ValueError(f"vision_map must be [B,C,H,W], got {tuple(vision_map.shape)}")
        batch, channels, height, width = vision_map.shape
        tokens = vision_map.flatten(2).transpose(1, 2)
        tokens = self.input_proj(tokens)
        coords = self._coordinates(height, width, tokens.device, tokens.dtype)
        tokens = tokens + self.position_proj(coords)
        queries = self.latent_queries.expand(batch, -1, -1)
        for block in self.blocks:
            queries = block(queries, tokens)
        return self.output_norm(queries)


class EntityDecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 3) -> None:
        super().__init__()
        self.entity_norm = nn.LayerNorm(dim)
        self.entity_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, entities: Tensor, memory: Tensor) -> Tensor:
        q = self.entity_norm(entities)
        update, _ = self.entity_attn(q, q, q, need_weights=False)
        entities = entities + update
        update, _ = self.cross_attn(
            self.cross_query_norm(entities),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        entities = entities + update
        return entities + self.ffn(self.ffn_norm(entities))


class EntityDecoder(nn.Module):
    def __init__(self, config: SpatialCoreConfig) -> None:
        super().__init__()
        dim = config.spatial_dim
        self.entity_queries = nn.Parameter(torch.randn(1, config.num_entities, dim) * 0.02)
        self.language_proj = nn.Linear(config.language_dim, dim)
        self.blocks = nn.ModuleList(
            [EntityDecoderBlock(dim, config.num_heads) for _ in range(config.entity_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.objectness = nn.Linear(dim, 1)
        self.box_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )

    def forward(self, spatial_tokens: Tensor, language_context: Tensor | None) -> dict[str, Tensor]:
        batch = spatial_tokens.shape[0]
        entities = self.entity_queries.expand(batch, -1, -1)
        if language_context is not None:
            if language_context.ndim == 3:
                language_context = language_context.mean(dim=1)
            entities = entities + self.language_proj(language_context).unsqueeze(1)
        for block in self.blocks:
            entities = block(entities, spatial_tokens)
        entities = self.norm(entities)

        # Predict center-x, center-y, width, height, then convert to ordered xyxy.
        raw = self.box_head(entities).sigmoid()
        center = raw[..., :2]
        size = raw[..., 2:].clamp_min(1e-4)
        top_left = (center - size * 0.5).clamp(0.0, 1.0)
        bottom_right = (center + size * 0.5).clamp(0.0, 1.0)
        boxes = torch.cat((torch.minimum(top_left, bottom_right), torch.maximum(top_left, bottom_right)), dim=-1)

        return {
            "entity_tokens": entities,
            "objectness_logits": self.objectness(entities).squeeze(-1),
            "boxes_xyxy_norm": boxes,
        }


class DetailFeatureStem(nn.Module):
    """Small RGB detail path; it is part of LSF-VLM, not an external vision model."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, rgb: Tensor) -> Tensor:
        return self.net(rgb)


class SpatialPixelDecoder(nn.Module):
    def __init__(self, config: SpatialCoreConfig) -> None:
        super().__init__()
        self.vision_proj = nn.Conv2d(config.vision_dim, config.pixel_dim, kernel_size=1)
        self.detail_stem = DetailFeatureStem(config.pixel_dim)
        self.fuse = nn.Sequential(
            nn.Conv2d(config.pixel_dim * 2, config.pixel_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, config.pixel_dim),
            nn.GELU(),
            nn.Conv2d(config.pixel_dim, config.pixel_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, config.pixel_dim),
            nn.GELU(),
        )
        self.mask_features = nn.Sequential(
            nn.Conv2d(config.pixel_dim, config.pixel_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.pixel_dim, config.pixel_dim, kernel_size=1),
        )
        self.entity_to_mask = nn.Linear(config.spatial_dim, config.pixel_dim)
        self.mask_bias = nn.Linear(config.spatial_dim, 1)

    def forward(self, vision_map: Tensor, rgb: Tensor, entity_tokens: Tensor) -> tuple[Tensor, Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"rgb must be [B,3,H,W], got {tuple(rgb.shape)}")
        target_size = (max(1, rgb.shape[-2] // 4), max(1, rgb.shape[-1] // 4))
        vision = self.vision_proj(vision_map)
        vision = F.interpolate(vision, size=target_size, mode="bilinear", align_corners=False)
        detail = self.detail_stem(rgb)
        if detail.shape[-2:] != target_size:
            detail = F.interpolate(detail, size=target_size, mode="bilinear", align_corners=False)
        pixel_features = self.fuse(torch.cat((vision, detail), dim=1))
        mask_features = F.normalize(self.mask_features(pixel_features), dim=1)
        mask_queries = self.entity_to_mask(entity_tokens)
        mask_logits = torch.einsum("bqd,bdhw->bqhw", mask_queries, mask_features)
        mask_logits = mask_logits + self.mask_bias(entity_tokens).unsqueeze(-1)
        return pixel_features, mask_logits


class DepthUncertaintyHead(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, 2, kernel_size=1),
        )

    def forward(self, pixel_features: Tensor, output_size: tuple[int, int]) -> tuple[Tensor, Tensor]:
        raw = self.net(pixel_features)
        raw = F.interpolate(raw, size=output_size, mode="bilinear", align_corners=False)
        relative_inverse_depth = F.softplus(raw[:, :1]) + 1e-5
        log_variance = raw[:, 1:2].clamp(-8.0, 8.0)
        return relative_inverse_depth, log_variance


class SpatialRelationHead(nn.Module):
    def __init__(self, dim: int, num_relations: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2 + 8, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_relations),
        )

    def forward(self, entities: Tensor, boxes: Tensor) -> Tensor:
        batch, count, dim = entities.shape
        subject = entities[:, :, None, :].expand(-1, -1, count, -1)
        object_ = entities[:, None, :, :].expand(-1, count, -1, -1)
        box_subject = boxes[:, :, None, :].expand(-1, -1, count, -1)
        box_object = boxes[:, None, :, :].expand(-1, count, -1, -1)
        delta = box_object - box_subject
        center_subject = (box_subject[..., :2] + box_subject[..., 2:]) * 0.5
        center_object = (box_object[..., :2] + box_object[..., 2:]) * 0.5
        center_delta = center_object - center_subject
        size_subject = (box_subject[..., 2:] - box_subject[..., :2]).clamp_min(1e-5)
        size_object = (box_object[..., 2:] - box_object[..., :2]).clamp_min(1e-5)
        size_ratio = torch.log(size_object / size_subject)
        geometry = torch.cat((delta, center_delta, size_ratio), dim=-1)
        pair = torch.cat((subject, object_, geometry), dim=-1)
        logits = self.net(pair)
        diagonal = torch.eye(count, device=entities.device, dtype=torch.bool).unsqueeze(0)
        return logits.masked_fill(diagonal.unsqueeze(-1), 0.0)


class LSFSpatialCore(nn.Module):
    """Shared spatial representation and unified entity-conditioned task heads.

    Inputs:
      vision_map: [B, vision_dim, patch_h, patch_w], pre-projector LFM3 features.
      rgb: [B, 3, image_h, image_w], RGB in [0, 1] or normalized consistently.
      language_context: optional [B, language_dim] or [B, text_length, language_dim].

    Outputs share entity indices across boxes, masks, and relation graph nodes.
    """

    def __init__(self, config: SpatialCoreConfig | None = None) -> None:
        super().__init__()
        self.config = config or SpatialCoreConfig()
        self.config.validate()
        self.spatial_bridge = PerceiverSpatialBridge(self.config)
        self.entity_decoder = EntityDecoder(self.config)
        self.pixel_decoder = SpatialPixelDecoder(self.config)
        self.depth_head = DepthUncertaintyHead(self.config.pixel_dim)
        self.relation_head = SpatialRelationHead(self.config.spatial_dim, self.config.num_relations)
        self.language_to_spatial = nn.Linear(self.config.language_dim, self.config.spatial_dim)
        self.language_phrase_to_spatial = nn.Linear(
            self.config.language_dim, self.config.spatial_dim, bias=False
        )
        self.entity_phrase_proj = nn.Linear(
            self.config.spatial_dim, self.config.spatial_dim, bias=False
        )
        self.language_norm = nn.LayerNorm(self.config.spatial_dim)

    def encode_scene(self, vision_map: Tensor) -> Tensor:
        return self.spatial_bridge(vision_map)

    def forward(
        self,
        vision_map: Tensor,
        rgb: Tensor,
        language_context: Tensor | None = None,
        spatial_tokens: Tensor | None = None,
        phrase_contexts: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if vision_map.shape[0] != rgb.shape[0]:
            raise ValueError("vision_map and rgb batch sizes must match")
        if language_context is not None and language_context.ndim == 3:
            language_context = language_context.mean(dim=1)
        if language_context is not None:
            language_context = self.language_norm(self.language_to_spatial(language_context))

        spatial_tokens = spatial_tokens if spatial_tokens is not None else self.encode_scene(vision_map)
        entities = self.entity_decoder(spatial_tokens, language_context)
        pixel_features, mask_logits = self.pixel_decoder(
            vision_map, rgb, entities["entity_tokens"]
        )
        depth, depth_log_variance = self.depth_head(
            pixel_features, output_size=rgb.shape[-2:]
        )
        relations = self.relation_head(
            entities["entity_tokens"], entities["boxes_xyxy_norm"]
        )
        phrase_logits = None
        if phrase_contexts is not None:
            if phrase_contexts.ndim != 3:
                raise ValueError("phrase_contexts must be [B,num_phrases,language_dim]")
            phrase_features = F.normalize(
                self.language_phrase_to_spatial(phrase_contexts), dim=-1
            )
            entity_features = F.normalize(
                self.entity_phrase_proj(entities["entity_tokens"]), dim=-1
            )
            phrase_logits = torch.einsum(
                "bqd,bpd->bqp", entity_features, phrase_features
            ) * (self.config.spatial_dim ** 0.5)
        result = {
            "spatial_tokens": spatial_tokens,
            **entities,
            "mask_logits": mask_logits,
            "relative_inverse_depth": depth,
            "depth_log_variance": depth_log_variance,
            "relation_logits": relations,
            "pixel_features": pixel_features,
        }
        if phrase_logits is not None:
            result["entity_phrase_logits"] = phrase_logits
        return result

    def parameter_report(self) -> dict[str, Any]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "config": asdict(self.config),
            "parameters_total": total,
            "parameters_trainable": trainable,
        }


def insert_soft_spatial_tokens(
    input_ids: Tensor,
    inputs_embeds: Tensor,
    attention_mask: Tensor,
    spatial_prefix: Tensor,
    insertion_index: int,
    placeholder_token_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Insert continuous spatial tokens into the LFM3 causal sequence.

    The placeholder IDs are bookkeeping only: the corresponding input embeddings
    are replaced with spatial_prefix before the language model forward pass.
    Insert after the image's vision_end token and before the user question whenever
    the standard LFM3 chat template is used.
    """
    if input_ids.ndim != 2 or inputs_embeds.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("Expected input_ids [B,L], inputs_embeds [B,L,D], attention_mask [B,L]")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask shapes must match")
    if inputs_embeds.shape[:2] != input_ids.shape:
        raise ValueError("inputs_embeds batch/sequence dimensions must match input_ids")
    if spatial_prefix.ndim != 3 or spatial_prefix.shape[0] != input_ids.shape[0]:
        raise ValueError("spatial_prefix must be [B,N,D] with matching batch size")
    if spatial_prefix.shape[-1] != inputs_embeds.shape[-1]:
        raise ValueError("spatial_prefix dimension must match language embedding dimension")
    if not 0 <= insertion_index <= input_ids.shape[1]:
        raise ValueError("insertion_index is outside the input sequence")

    batch, count, _ = spatial_prefix.shape
    placeholder_ids = torch.full(
        (batch, count),
        fill_value=placeholder_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    prefix_mask = torch.ones(
        (batch, count), dtype=attention_mask.dtype, device=attention_mask.device
    )
    new_ids = torch.cat(
        (input_ids[:, :insertion_index], placeholder_ids, input_ids[:, insertion_index:]),
        dim=1,
    )
    new_embeds = torch.cat(
        (inputs_embeds[:, :insertion_index], spatial_prefix, inputs_embeds[:, insertion_index:]),
        dim=1,
    )
    new_mask = torch.cat(
        (attention_mask[:, :insertion_index], prefix_mask, attention_mask[:, insertion_index:]),
        dim=1,
    )
    return new_ids, new_embeds, new_mask


def find_vision_end_insertion_index(input_ids: Tensor, vision_end_token_id: int) -> int:
    """Find the first image closing marker for a single-image LFM3 prompt."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("The initial integration supports batch_size=1")
    matches = torch.nonzero(input_ids[0] == vision_end_token_id, as_tuple=False).flatten()
    if matches.numel() == 0:
        raise ValueError("No vision_end token found; verify the LFM3 chat template")
    return int(matches[0].item()) + 1
