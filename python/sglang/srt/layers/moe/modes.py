# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Runtime MoDES route masking for MoE top-k outputs.

This first integration is intentionally conservative: it operates on standard
top-k tensors after routing has selected experts and before the MoE runner sees
the route list. It is meant for text-only experiments on Qwen-style MoE models.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModesRuntimeConfig:
    enabled: bool
    alpha_path: Optional[str]
    tau_text: float
    min_experts_per_token: int
    force_standard_topk: bool


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@lru_cache(maxsize=1)
def get_modes_runtime_config() -> ModesRuntimeConfig:
    return ModesRuntimeConfig(
        enabled=_env_bool("SGLANG_ENABLE_MODES", False),
        alpha_path=os.getenv("SGLANG_MODES_ALPHA_PATH"),
        tau_text=float(os.getenv("SGLANG_MODES_TAU_TEXT", "0") or 0.0),
        min_experts_per_token=max(
            0, int(os.getenv("SGLANG_MODES_MIN_EXPERTS_PER_TOKEN", "0") or 0)
        ),
        force_standard_topk=_env_bool("SGLANG_MODES_FORCE_STANDARD_TOPK", True),
    )


def modes_is_enabled() -> bool:
    return get_modes_runtime_config().enabled


def modes_force_standard_topk() -> bool:
    config = get_modes_runtime_config()
    return config.enabled and config.force_standard_topk


_PATCH_INSTALLED = False


def install_modes_patches() -> None:
    """Install env-gated MoDES patches for SGLang MoE routing.

    The patch keeps the public launch surface small for the first integration:
    setting ``SGLANG_ENABLE_MODES=1`` makes TopK materialize explicit routing
    tensors and applies MoDES masking immediately after standard TopK routing.
    """

    global _PATCH_INSTALLED
    if _PATCH_INSTALLED or not modes_is_enabled():
        return

    from sglang.srt.layers.moe import topk as topk_module

    if modes_force_standard_topk():
        original_init = topk_module.TopK.__init__

        def patched_topk_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.topk_config.output_format = topk_module.TopKOutputFormat.STANDARD

        topk_module.TopK.__init__ = patched_topk_init

    original_select_experts = topk_module.select_experts

    def patched_select_experts(*args, **kwargs):
        output = original_select_experts(*args, **kwargs)
        topk_weights, topk_ids = apply_modes_to_topk(
            layer_id=kwargs.get("layer_id"),
            topk_weights=output.topk_weights,
            topk_ids=output.topk_ids,
            num_fused_shared_experts=kwargs["topk_config"].num_fused_shared_experts,
        )
        return output._replace(topk_weights=topk_weights, topk_ids=topk_ids)

    topk_module.select_experts = patched_select_experts
    _PATCH_INSTALLED = True
    logger.info("Installed experimental MoDES MoE route masking patches.")


@lru_cache(maxsize=8)
def _load_alpha(alpha_path: Optional[str]) -> Optional[tuple[float, ...]]:
    if not alpha_path:
        return None

    with open(alpha_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    alpha = obj.get("alpha") if isinstance(obj, dict) else obj
    if not isinstance(alpha, list):
        raise ValueError(
            f"MoDES alpha file must be a list or contain an 'alpha' list: {alpha_path}"
        )

    values = tuple(float(x) for x in alpha)
    if not values:
        raise ValueError(f"MoDES alpha list is empty: {alpha_path}")
    return values


def _alpha_for_layer(layer_id: Optional[int], alpha_path: Optional[str]) -> float:
    alpha = _load_alpha(alpha_path)
    if alpha is None or layer_id is None:
        return 1.0
    if layer_id < 0:
        return 1.0
    if layer_id >= len(alpha):
        return alpha[-1]
    return alpha[layer_id]


def _protect_min_experts(
    skip_mask: torch.Tensor,
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    min_experts_per_token: int,
) -> torch.Tensor:
    if min_experts_per_token <= 0 or scores.shape[1] <= min_experts_per_token:
        return skip_mask

    num_valid = valid_mask.sum(dim=-1, keepdim=True)
    min_keep = torch.clamp(
        torch.full_like(num_valid, min_experts_per_token), max=scores.shape[1]
    )
    protected_scores = scores.masked_fill(~valid_mask, float("-inf"))
    keep_ids = torch.topk(
        protected_scores,
        k=min_experts_per_token,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices
    protect_mask = torch.zeros_like(skip_mask)
    protect_mask.scatter_(1, keep_ids, True)
    protect_mask = protect_mask & (num_valid >= min_keep)
    return skip_mask & ~protect_mask


def apply_modes_to_topk(
    *,
    layer_id: Optional[int],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_fused_shared_experts: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MoDES masking to routed expert slots.

    A route is masked when ``topk_weight * alpha[layer_id] < tau_text``. Masked
    routes get weight 0 while keeping their original expert id. Keeping expert
    ids valid avoids Triton fused-MoE kernels that do not accept ``-1`` in
    active routed slots.
    """

    config = get_modes_runtime_config()
    if not config.enabled or config.tau_text <= 0.0 or topk_weights.numel() == 0:
        return topk_weights, topk_ids

    routed_cols = topk_weights.shape[1] - max(0, num_fused_shared_experts)
    if routed_cols <= 0:
        return topk_weights, topk_ids

    routed_weights = topk_weights[:, :routed_cols]
    routed_ids = topk_ids[:, :routed_cols]
    alpha = _alpha_for_layer(layer_id, config.alpha_path)

    scores = routed_weights.float() * alpha
    valid_mask = routed_ids >= 0
    skip_mask = (scores < config.tau_text) & valid_mask
    skip_mask = _protect_min_experts(
        skip_mask, scores, valid_mask, config.min_experts_per_token
    )

    topk_weights = topk_weights.clone()
    topk_ids = topk_ids.clone()
    topk_weights[:, :routed_cols] = routed_weights.masked_fill(skip_mask, 0.0)
    return topk_weights, topk_ids
