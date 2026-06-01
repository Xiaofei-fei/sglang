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
import tempfile
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModesRuntimeConfig:
    enabled: bool
    alpha_path: Optional[str]
    tau_text: Optional[float]
    auto_tau: bool
    target_skip_rate: float
    calibration_routes: int
    min_experts_per_token: int
    force_standard_topk: bool
    metrics_path: Optional[str]
    metrics_flush_interval: int


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_float_optional(name: str) -> Optional[float]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return float(value)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@lru_cache(maxsize=1)
def get_modes_runtime_config() -> ModesRuntimeConfig:
    tau_text = _env_float_optional("SGLANG_MODES_TAU_TEXT")
    return ModesRuntimeConfig(
        enabled=_env_bool("SGLANG_ENABLE_MODES", False),
        alpha_path=os.getenv("SGLANG_MODES_ALPHA_PATH"),
        tau_text=tau_text,
        auto_tau=_env_bool("SGLANG_MODES_AUTO_TAU", tau_text is None),
        target_skip_rate=_clamp(
            float(os.getenv("SGLANG_MODES_TARGET_SKIP_RATE", "0.13") or 0.13),
            0.0,
            1.0,
        ),
        calibration_routes=max(
            1, int(os.getenv("SGLANG_MODES_CALIBRATION_ROUTES", "500000") or 500000)
        ),
        min_experts_per_token=max(
            0, int(os.getenv("SGLANG_MODES_MIN_EXPERTS_PER_TOKEN", "0") or 0)
        ),
        force_standard_topk=_env_bool("SGLANG_MODES_FORCE_STANDARD_TOPK", True),
        metrics_path=os.getenv("SGLANG_MODES_METRICS_PATH"),
        metrics_flush_interval=max(
            1, int(os.getenv("SGLANG_MODES_METRICS_FLUSH_INTERVAL", "100") or 100)
        ),
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


def _atomic_write_json(path: str, obj: dict) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".modes-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class _ModesRuntimeState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.score_chunks: list[torch.Tensor] = []
        self.calibration_route_count = 0
        self.calibrated_tau_text: Optional[float] = None
        self.calls = 0
        self.route_count = 0
        self.skip_count = 0
        self.active_route_count = 0
        self.active_skip_count = 0

    def effective_tau_text(self, config: ModesRuntimeConfig) -> Optional[float]:
        if config.tau_text is not None and config.tau_text > 0.0:
            return config.tau_text
        if not config.auto_tau:
            return None
        with self.lock:
            return self.calibrated_tau_text

    def observe_scores_for_auto_tau(
        self, scores: torch.Tensor, config: ModesRuntimeConfig
    ) -> None:
        if config.tau_text is not None or not config.auto_tau:
            return
        if scores.numel() == 0:
            return

        score_chunk = scores.detach().float().flatten().cpu()
        with self.lock:
            if self.calibrated_tau_text is not None:
                return
            remaining = config.calibration_routes - self.calibration_route_count
            if remaining <= 0:
                return

            if score_chunk.numel() > remaining:
                score_chunk = score_chunk[:remaining]
            self.score_chunks.append(score_chunk)
            self.calibration_route_count += int(score_chunk.numel())

            if self.calibration_route_count < config.calibration_routes:
                return

            all_scores = torch.cat(self.score_chunks)
            if config.target_skip_rate <= 0.0:
                tau_text = 0.0
            elif config.target_skip_rate >= 1.0:
                tau_text = float(all_scores.max().item())
            else:
                tau_text = float(
                    torch.quantile(all_scores, config.target_skip_rate).item()
                )
            self.calibrated_tau_text = tau_text
            self.score_chunks.clear()

        logger.info(
            "MoDES auto-calibrated tau_text=%s from %s routed scores "
            "for target_skip_rate=%s.",
            tau_text,
            config.calibration_routes,
            config.target_skip_rate,
        )
        self.maybe_write_metrics(config, force=True)

    def record_routes(
        self,
        route_count: int,
        skip_count: int,
        config: ModesRuntimeConfig,
        *,
        masking_active: bool,
    ) -> None:
        with self.lock:
            self.calls += 1
            self.route_count += route_count
            self.skip_count += skip_count
            if masking_active:
                self.active_route_count += route_count
                self.active_skip_count += skip_count
            should_flush = self.calls % config.metrics_flush_interval == 0
        if should_flush:
            self.maybe_write_metrics(config)

    def maybe_write_metrics(
        self, config: ModesRuntimeConfig, *, force: bool = False
    ) -> None:
        if not config.metrics_path:
            return
        with self.lock:
            route_count = self.route_count
            skip_count = self.skip_count
            active_route_count = self.active_route_count
            active_skip_count = self.active_skip_count
            obj = {
                "active_route_count": active_route_count,
                "active_skip_count": active_skip_count,
                "active_skip_rate": active_skip_count / active_route_count
                if active_route_count
                else 0.0,
                "auto_tau": config.auto_tau and config.tau_text is None,
                "calibrated": self.calibrated_tau_text is not None
                or (config.tau_text is not None and config.tau_text > 0.0),
                "calibration_route_count": self.calibration_route_count,
                "calibration_routes": config.calibration_routes,
                "calls": self.calls,
                "configured_tau_text": config.tau_text,
                "effective_tau_text": self.calibrated_tau_text
                if config.tau_text is None
                else config.tau_text,
                "route_count": route_count,
                "skip_count": skip_count,
                "skip_rate": skip_count / route_count if route_count else 0.0,
                "target_skip_rate": config.target_skip_rate,
            }

        try:
            _atomic_write_json(config.metrics_path, obj)
        except Exception:
            logger.exception("Failed to write MoDES metrics to %s", config.metrics_path)


_MODES_STATE = _ModesRuntimeState()


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
    if not config.enabled or topk_weights.numel() == 0:
        return topk_weights, topk_ids

    routed_cols = topk_weights.shape[1] - max(0, num_fused_shared_experts)
    if routed_cols <= 0:
        return topk_weights, topk_ids

    routed_weights = topk_weights[:, :routed_cols]
    routed_ids = topk_ids[:, :routed_cols]
    alpha = _alpha_for_layer(layer_id, config.alpha_path)

    scores = routed_weights.float() * alpha
    valid_mask = routed_ids >= 0
    valid_scores = scores[valid_mask]
    tau_text = _MODES_STATE.effective_tau_text(config)

    if tau_text is None or tau_text <= 0.0:
        _MODES_STATE.observe_scores_for_auto_tau(valid_scores, config)
        if config.metrics_path:
            _MODES_STATE.record_routes(
                int(valid_mask.sum().item()), 0, config, masking_active=False
            )
        return topk_weights, topk_ids

    skip_mask = (scores < tau_text) & valid_mask
    skip_mask = _protect_min_experts(
        skip_mask, scores, valid_mask, config.min_experts_per_token
    )
    if config.metrics_path:
        _MODES_STATE.record_routes(
            int(valid_mask.sum().item()),
            int(skip_mask.sum().item()),
            config,
            masking_active=True,
        )

    topk_weights = topk_weights.clone()
    topk_ids = topk_ids.clone()
    topk_weights[:, :routed_cols] = routed_weights.masked_fill(skip_mask, 0.0)
    return topk_weights, topk_ids
