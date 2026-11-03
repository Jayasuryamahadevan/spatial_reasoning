from types import SimpleNamespace

import torch
from torch import nn

from lsf_vlm.modeling.lfm3_adapter import UnifiedLSFVLM
from lsf_vlm.modeling.spatial_core import SpatialCoreConfig


class FakeVisionTower(nn.Module):
    def __init__(self, vision_dim: int) -> None:
        super().__init__()
        self.patch_projection = nn.Linear(1, vision_dim, bias=False)
        self.calls = 0

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.patch_projection(pixel_values.reshape(-1, 1))


class FakeLFM3Model(nn.Module):
    def __init__(self, vision_dim: int, language_dim: int, image_token_id: int) -> None:
        super().__init__()
        self.vision_tower = FakeVisionTower(vision_dim)
        self.embedding = nn.Embedding(32, language_dim)
        self.projector = nn.Linear(vision_dim, language_dim, bias=False)
        self.language = nn.Linear(language_dim, language_dim, bias=False)
        self.image_token_id = image_token_id

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        patch_features = self.vision_tower(pixel_values)
        height, width = int(image_grid_thw[0, 1]), int(image_grid_thw[0, 2])
        grid = patch_features.reshape(height, width, -1)
        merged = grid.reshape(height // 2, 2, width // 2, 2, -1).mean(dim=(1, 3))
        projected = self.projector(merged.reshape(-1, merged.shape[-1]))
        hidden_states = (
            patch_features * 0.7,
            patch_features * 0.8,
            patch_features * 0.9,
            patch_features,
        )
        return SimpleNamespace(
            last_hidden_state=patch_features,
            hidden_states=hidden_states,
            pooler_output=(projected,),
        )

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        mm_encoder_outputs=None,
        use_cache=False,
        return_dict=True,
        output_hidden_states=False,
    ) -> SimpleNamespace:
        assert input_ids is None
        assert mm_encoder_outputs is not None
        image_features = torch.cat(mm_encoder_outputs["image"].pooler_output, dim=0)
        image_embedding = self.embedding(
            torch.tensor(self.image_token_id, device=inputs_embeds.device)
        )
        image_mask = (inputs_embeds == image_embedding).all(dim=-1)
        assert int(image_mask.sum()) == image_features.shape[0]
        expanded_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(expanded_mask, image_features.reshape(-1))
        hidden = self.language(inputs_embeds)
        return SimpleNamespace(
            last_hidden_state=hidden,
            past_key_values=None,
        )


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(hidden_size=64),
            text_config=SimpleNamespace(hidden_size=128),
            vision_end_token_id=3,
            pad_token_id=0,
        )
        self.model = FakeLFM3Model(64, 128, image_token_id=2)
        self.lm_head = nn.Linear(128, 32, bias=False)


def test_unified_adapter_uses_one_vision_pass_and_emits_all_spatial_heads():
    torch.manual_seed(31)
    backbone = FakeBackbone()
    model = UnifiedLSFVLM(
        backbone,
        spatial_config=SpatialCoreConfig(
            vision_dim=64,
            language_dim=128,
            spatial_dim=64,
            num_spatial_latents=64,
            num_entities=8,
            num_heads=4,
            perceiver_layers=1,
            entity_layers=1,
            pixel_dim=32,
            num_relations=10,
        ),
        freeze_backbone=True,
    ).eval()

    input_ids = torch.tensor([[1, 2, 2, 2, 2, 3, 4, 5, 6, 7]])
    attention_mask = torch.ones_like(input_ids)
    image_grid_thw = torch.tensor([[1, 4, 4]])
    pixel_values = torch.rand(16, 1)
    rgb = torch.rand(1, 3, 64, 64)

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            rgb=rgb,
            compute_language_logits=True,
            use_cache=False,
        )

    assert backbone.model.vision_tower.calls == 1
    assert output["spatial_tokens"].shape == (1, 64, 64)
    assert output["entity_tokens"].shape == (1, 8, 64)
    assert output["boxes_xyxy_norm"].shape == (1, 8, 4)
    assert output["mask_logits"].shape == (1, 8, 16, 16)
    assert output["relative_depth"].shape == (1, 1, 64, 64)
    assert output["relation_logits"].shape == (1, 8, 8, 10)
    assert output["language_logits"].shape[1] == input_ids.shape[1] + 64
    assert output["expanded_input_ids"].shape[1] == input_ids.shape[1] + 64
