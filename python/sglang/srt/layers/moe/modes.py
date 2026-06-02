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
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModesRuntimeConfig:
    enabled: bool
    config_path: Optional[str]
    save_config_path: Optional[str]
    alpha_path: Optional[str]
    auto_alpha: bool
    alpha_min: float
    alpha_max: float
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


def _extract_config_payload(obj: dict[str, Any]) -> dict[str, Any]:
    """Return the part of a MoDES JSON file that contains runtime knobs."""

    selected = obj.get("selected_config")
    if isinstance(selected, dict):
        return selected

    best = obj.get("best")
    if isinstance(best, dict) and isinstance(best.get("config"), dict):
        return best["config"]

    config = obj.get("config")
    if isinstance(config, dict) and (
        "alpha" in config or "tau_text" in config or "effective_tau_text" in config
    ):
        return config

    return obj


@lru_cache(maxsize=8)
def _load_modes_config(config_path: Optional[str]) -> Optional[dict[str, Any]]:
    if not config_path:
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"MoDES config must be a JSON object: {config_path}")
    return _extract_config_payload(obj)


@lru_cache(maxsize=8)
def _load_config_alpha(config_path: Optional[str]) -> Optional[tuple[float, ...]]:
    obj = _load_modes_config(config_path)
    if obj is None:
        return None

    alpha = obj.get("alpha")
    if alpha is None:
        return None
    if not isinstance(alpha, list):
        raise ValueError(f"MoDES config alpha must be a JSON list: {config_path}")

    values = tuple(float(x) for x in alpha)
    if not values:
        raise ValueError(f"MoDES config alpha list is empty: {config_path}")
    return values


def _config_float(
    config_path: Optional[str], keys: tuple[str, ...]
) -> Optional[float]:
    obj = _load_modes_config(config_path)
    if obj is None:
        return None

    for key in keys:
        value = obj.get(key)
        if value is not None:
            return float(value)
    return None


def _env_float_with_config_default(
    name: str, config_path: Optional[str], config_keys: tuple[str, ...], default: float
) -> float:
    value = os.getenv(name)
    if value is not None and value.strip() != "":
        return float(value)

    config_value = _config_float(config_path, config_keys)
    if config_value is not None:
        return config_value
    return default


def _env_int_with_config_default(
    name: str, config_path: Optional[str], config_keys: tuple[str, ...], default: int
) -> int:
    value = os.getenv(name)
    if value is not None and value.strip() != "":
        return int(value)

    config_value = _config_float(config_path, config_keys)
    if config_value is not None:
        return int(config_value)
    return default


@lru_cache(maxsize=1)
def get_modes_runtime_config() -> ModesRuntimeConfig:
    config_path = os.getenv("SGLANG_MODES_CONFIG")
    alpha_path = os.getenv("SGLANG_MODES_ALPHA_PATH")
    config_alpha = _load_config_alpha(config_path)
    config_tau_text = _config_float(
        config_path, ("tau_text", "effective_tau_text", "configured_tau_text")
    )
    tau_text = (
        config_tau_text
        if config_tau_text is not None
        else _env_float_optional("SGLANG_MODES_TAU_TEXT")
    )
    return ModesRuntimeConfig(
        enabled=_env_bool("SGLANG_ENABLE_MODES", False),
        config_path=config_path,
        save_config_path=os.getenv("SGLANG_MODES_SAVE_CONFIG"),
        alpha_path=alpha_path,
        auto_alpha=_env_bool(
            "SGLANG_MODES_AUTO_ALPHA",
            alpha_path is None and config_alpha is None,
        ),
        alpha_min=max(
            0.0,
            _env_float_with_config_default(
                "SGLANG_MODES_ALPHA_MIN", config_path, ("alpha_min",), 0.25
            ),
        ),
        alpha_max=max(
            0.0,
            _env_float_with_config_default(
                "SGLANG_MODES_ALPHA_MAX", config_path, ("alpha_max",), 4.0
            ),
        ),
        tau_text=tau_text,
        auto_tau=_env_bool("SGLANG_MODES_AUTO_TAU", tau_text is None),
        target_skip_rate=_clamp(
            _env_float_with_config_default(
                "SGLANG_MODES_TARGET_SKIP_RATE",
                config_path,
                ("target_skip_rate", "active_skip_rate"),
                0.13,
            ),
            0.0,
            1.0,
        ),
        calibration_routes=max(
            1,
            _env_int_with_config_default(
                "SGLANG_MODES_CALIBRATION_ROUTES",
                config_path,
                ("calibration_routes",),
                500000,
            ),
        ),
        min_experts_per_token=max(
            0,
            _env_int_with_config_default(
                "SGLANG_MODES_MIN_EXPERTS_PER_TOKEN",
                config_path,
                ("min_experts_per_token",),
                0,
            ),
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


def _alpha_from_values(
    layer_id: Optional[int], alpha: Optional[tuple[float, ...]]
) -> float:
    if alpha is None or layer_id is None or layer_id < 0:
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
        self.route_chunks: list[tuple[int, torch.Tensor]] = []
        self.calibration_route_count = 0
        self.calibrated_alpha: Optional[tuple[float, ...]] = None
        self.calibrated_tau_text: Optional[float] = None
        self.saved_config = False
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

    def effective_alpha_for_layer(
        self, layer_id: Optional[int], config: ModesRuntimeConfig
    ) -> float:
        config_alpha = _load_config_alpha(config.config_path)
        if config_alpha is not None:
            return _alpha_from_values(layer_id, config_alpha)
        if config.alpha_path:
            return _alpha_for_layer(layer_id, config.alpha_path)
        with self.lock:
            calibrated_alpha = self.calibrated_alpha
        return _alpha_from_values(layer_id, calibrated_alpha)

    def alpha_is_ready(self, config: ModesRuntimeConfig) -> bool:
        if _load_config_alpha(config.config_path) is not None:
            return True
        if config.alpha_path or not config.auto_alpha:
            return True
        with self.lock:
            return self.calibrated_alpha is not None

    def observe_routes_for_auto_calibration(
        self,
        *,
        layer_id: Optional[int],
        raw_scores: torch.Tensor,
        config: ModesRuntimeConfig,
    ) -> None:
        needs_auto_alpha = (
            config.auto_alpha
            and config.alpha_path is None
            and _load_config_alpha(config.config_path) is None
        )
        needs_auto_tau = config.tau_text is None and config.auto_tau
        if not needs_auto_alpha and not needs_auto_tau:
            return
        if raw_scores.numel() == 0:
            return

        score_chunk = raw_scores.detach().float().flatten().cpu()
        layer_index = int(layer_id) if layer_id is not None and layer_id >= 0 else -1
        with self.lock:
            if self.calibrated_tau_text is not None:
                return
            remaining = config.calibration_routes - self.calibration_route_count
            if remaining <= 0:
                return

            if score_chunk.numel() > remaining:
                score_chunk = score_chunk[:remaining]
            self.route_chunks.append((layer_index, score_chunk))
            self.calibration_route_count += int(score_chunk.numel())

            if self.calibration_route_count < config.calibration_routes:
                return

            calibrated_alpha: Optional[tuple[float, ...]] = None
            if needs_auto_alpha:
                layer_sums: dict[int, float] = {}
                layer_counts: dict[int, int] = {}
                total_sum = 0.0
                total_count = 0
                for chunk_layer_id, chunk in self.route_chunks:
                    if chunk_layer_id < 0:
                        continue
                    chunk_sum = float(chunk.sum().item())
                    chunk_count = int(chunk.numel())
                    layer_sums[chunk_layer_id] = (
                        layer_sums.get(chunk_layer_id, 0.0) + chunk_sum
                    )
                    layer_counts[chunk_layer_id] = (
                        layer_counts.get(chunk_layer_id, 0) + chunk_count
                    )
                    total_sum += chunk_sum
                    total_count += chunk_count

                if total_count > 0 and layer_sums:
                    global_mean = total_sum / total_count
                    max_layer = max(layer_sums)
                    alpha_values = []
                    eps = 1e-12
                    for idx in range(max_layer + 1):
                        count = layer_counts.get(idx, 0)
                        if count == 0:
                            alpha_values.append(1.0)
                            continue
                        layer_mean = layer_sums[idx] / count
                        alpha_values.append(
                            _clamp(
                                global_mean / (layer_mean + eps),
                                config.alpha_min,
                                config.alpha_max,
                            )
                        )
                    calibrated_alpha = tuple(float(x) for x in alpha_values)
                    self.calibrated_alpha = calibrated_alpha

            score_chunks = []
            for chunk_layer_id, chunk in self.route_chunks:
                if config.alpha_path:
                    alpha = _alpha_for_layer(chunk_layer_id, config.alpha_path)
                elif calibrated_alpha is not None:
                    alpha = _alpha_from_values(chunk_layer_id, calibrated_alpha)
                else:
                    alpha = 1.0
                score_chunks.append(chunk * alpha)

            all_scores = torch.cat(score_chunks)
            if config.target_skip_rate <= 0.0:
                tau_text = 0.0
            elif config.target_skip_rate >= 1.0:
                tau_text = float(all_scores.max().item())
            else:
                tau_text = float(
                    torch.quantile(all_scores, config.target_skip_rate).item()
                )
            self.calibrated_tau_text = tau_text
            self.route_chunks.clear()

        logger.info(
            "MoDES auto-calibrated alpha=%s tau_text=%s from %s routed scores "
            "for target_skip_rate=%s.",
            "on" if calibrated_alpha is not None else "off",
            tau_text,
            config.calibration_routes,
            config.target_skip_rate,
        )
        self.maybe_save_config(config)
        self.maybe_write_metrics(config, force=True)

    def maybe_save_config(self, config: ModesRuntimeConfig) -> None:
        if not config.save_config_path:
            return
        config_alpha = _load_config_alpha(config.config_path)

        with self.lock:
            if self.saved_config:
                return
            calibrated_alpha = self.calibrated_alpha
            calibrated_tau_text = self.calibrated_tau_text
            active_route_count = self.active_route_count
            active_skip_count = self.active_skip_count

        tau_text = config.tau_text if config.tau_text is not None else calibrated_tau_text
        if tau_text is None:
            return

        alpha = calibrated_alpha
        if alpha is None:
            alpha = config_alpha
        if alpha is None and config.alpha_path:
            alpha = _load_alpha(config.alpha_path)
        effective_auto_alpha = (
            config.auto_alpha and config.alpha_path is None and config_alpha is None
        )

        obj: dict[str, Any] = {
            "version": 1,
            "alpha": list(alpha) if alpha is not None else None,
            "tau_text": tau_text,
            "target_skip_rate": config.target_skip_rate,
            "calibration_routes": config.calibration_routes,
            "alpha_min": config.alpha_min,
            "alpha_max": config.alpha_max,
            "min_experts_per_token": config.min_experts_per_token,
            "source": {
                "config_path": config.config_path,
                "alpha_path": config.alpha_path,
                "auto_alpha": effective_auto_alpha,
                "auto_tau": config.auto_tau,
            },
            "metrics": {
                "active_route_count": active_route_count,
                "active_skip_count": active_skip_count,
                "active_skip_rate": active_skip_count / active_route_count
                if active_route_count
                else 0.0,
            },
        }

        try:
            _atomic_write_json(config.save_config_path, obj)
        except Exception:
            logger.exception(
                "Failed to save MoDES runtime config to %s",
                config.save_config_path,
            )
            return

        with self.lock:
            self.saved_config = True
        logger.info("Saved MoDES runtime config to %s.", config.save_config_path)

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
        config_alpha = _load_config_alpha(config.config_path)
        effective_auto_alpha = (
            config.auto_alpha and config.alpha_path is None and config_alpha is None
        )
        with self.lock:
            route_count = self.route_count
            skip_count = self.skip_count
            active_route_count = self.active_route_count
            active_skip_count = self.active_skip_count
            calibrated_alpha = self.calibrated_alpha
            obj = {
                "active_route_count": active_route_count,
                "active_skip_count": active_skip_count,
                "active_skip_rate": active_skip_count / active_route_count
                if active_route_count
                else 0.0,
                "alpha_max": config.alpha_max,
                "alpha_min": config.alpha_min,
                "auto_alpha": effective_auto_alpha,
                "auto_tau": config.auto_tau and config.tau_text is None,
                "config_loaded": config.config_path is not None,
                "config_path": config.config_path,
                "calibrated_alpha": list(calibrated_alpha)
                if calibrated_alpha is not None
                else None,
                "calibrated": self.calibrated_tau_text is not None
                or (config.tau_text is not None and config.tau_text > 0.0),
                "calibration_route_count": self.calibration_route_count,
                "calibration_routes": config.calibration_routes,
                "calls": self.calls,
                "configured_tau_text": config.tau_text,
                "effective_tau_text": self.calibrated_tau_text
                if config.tau_text is None
                else config.tau_text,
                "loaded_alpha": list(config_alpha)
                if config_alpha is not None
                else None,
                "route_count": route_count,
                "save_config_path": config.save_config_path,
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
    raw_scores = routed_weights.float()
    valid_mask = routed_ids >= 0
    valid_raw_scores = raw_scores[valid_mask]

    if not _MODES_STATE.alpha_is_ready(config):
        _MODES_STATE.observe_routes_for_auto_calibration(
            layer_id=layer_id, raw_scores=valid_raw_scores, config=config
        )
        if config.metrics_path:
            _MODES_STATE.record_routes(
                int(valid_mask.sum().item()), 0, config, masking_active=False
            )
        return topk_weights, topk_ids

    alpha = _MODES_STATE.effective_alpha_for_layer(layer_id, config)
    scores = raw_scores * alpha
    tau_text = _MODES_STATE.effective_tau_text(config)

    if tau_text is None or tau_text <= 0.0:
        _MODES_STATE.observe_routes_for_auto_calibration(
            layer_id=layer_id, raw_scores=valid_raw_scores, config=config
        )
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
