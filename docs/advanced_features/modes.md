# Experimental MoDES Route Masking

This fork includes an experimental text-only MoDES integration for MoE models.
It masks low-importance routed expert slots after top-k routing and before the
MoE runner consumes the route list.

## Launch

By default, MoDES auto-calibrates both layer `alpha` and `tau_text` from the
live routed-score distribution. The minimal launch only needs the feature
switch:

```bash
python -m sglang.launch_server \
  --model-path /models/pubulic-models/Qwen3.5-35B-A3B-FP8 \
  --moe-runner-backend triton \
  --enable-modes
```

For debuggability, metrics can be enabled explicitly:

```bash
python -m sglang.launch_server \
  --model-path /models/pubulic-models/Qwen3.5-35B-A3B-FP8 \
  --moe-runner-backend triton \
  --enable-modes \
  --modes-metrics-path /tmp/modes_metrics.json
```

To reuse the same model/dataset calibration later, save the auto-calibrated
runtime config during the first validated run:

```bash
python -m sglang.launch_server \
  --model-path /models/pubulic-models/Qwen3.5-35B-A3B-FP8 \
  --moe-runner-backend triton \
  --enable-modes \
  --modes-save-config /tmp/qwen35b_mmlu_pro_modes.json \
  --modes-metrics-path /tmp/modes_metrics.json
```

Then load the saved config directly:

```bash
python -m sglang.launch_server \
  --model-path /models/pubulic-models/Qwen3.5-35B-A3B-FP8 \
  --moe-runner-backend triton \
  --enable-modes \
  --modes-config /tmp/qwen35b_mmlu_pro_modes.json
```

If neither `SGLANG_MODES_ALPHA_PATH` nor `SGLANG_MODES_TAU_TEXT` is set, the
first `SGLANG_MODES_CALIBRATION_ROUTES` routed scores are used as a calibration
window. SGLang estimates:

```text
alpha[layer] = global_mean(topk_weight) / layer_mean(topk_weight)
```

and then chooses:

```text
score    = topk_weight * alpha[layer]
tau_text = quantile(score, SGLANG_MODES_TARGET_SKIP_RATE)
```

and starts masking routes after the calibration window. For stable benchmark
numbers, run a short warmup first and exclude calibration requests from the
reported latency/throughput.

For fully fixed experiments, set `SGLANG_MODES_TAU_TEXT` explicitly:

```bash
python -m sglang.launch_server \
  --model-path /models/pubulic-models/Qwen3.5-35B-A3B-FP8 \
  --moe-runner-backend triton \
  --enable-modes \
  --modes-alpha-path /path/to/modes_alpha.json \
  --modes-tau-text 0.000438 \
  --modes-min-experts-per-token 0
```

For validated reuse, prefer `--modes-config` over separate alpha/tau settings.
The config file can be the direct output from `--modes-save-config` or the
output from `modes_select_config`.

The alpha file can either be a raw JSON list:

```json
[0.01, 0.02, 0.03]
```

or an object with an `alpha` list:

```json
{"alpha": [0.01, 0.02, 0.03]}
```

## Runtime Behavior

For each routed expert slot:

```text
score = topk_weight * alpha[layer_id]
skip  = score < tau_text
```

Masked slots are written as:

```text
topk_weight = 0
topk_id = original expert id
```

The expert id is intentionally kept valid because the Triton fused-MoE path does
not accept `-1` inside active routed slots. The integration is only applied to
standard top-k outputs. When MoDES is enabled, SGLang forces standard top-k by
default so the mask can be applied outside fused top-k kernels.

Set `SGLANG_MODES_FORCE_STANDARD_TOPK=0` to disable this behavior.

## CLI Arguments

Passing any `--modes-*` argument implicitly enables MoDES. `--enable-modes` is
still recommended in benchmark commands because it makes the experiment state
obvious.

| Argument | Description |
|---|---|
| `--enable-modes` | Enables MoDES route masking. |
| `--modes-config` | Loads a reusable MoDES config containing `alpha` and `tau_text`. |
| `--modes-save-config` | Saves the effective auto-calibrated config once alpha/tau are ready. |
| `--modes-alpha-path` | Path to a JSON alpha list or `{"alpha": [...]}` object. |
| `--modes-tau-text` | Fixed text threshold. When unset, auto tau is used. |
| `--modes-target-skip-rate` | Target routed-slot skip rate used by auto tau. |
| `--modes-calibration-routes` | Number of routed scores collected before auto calibration is finalized. |
| `--modes-min-experts-per-token` | Minimum routed experts kept per token after masking. |
| `--modes-metrics-path` | Optional JSON metrics output path. |
| `--modes-metrics-flush-interval` | Number of MoE calls between metrics writes. |
| `--modes-alpha-min` | Lower clamp for auto alpha. |
| `--modes-alpha-max` | Upper clamp for auto alpha. |
| `--modes-disable-auto-alpha` | Disables auto alpha calibration. |
| `--modes-disable-auto-tau` | Disables auto tau calibration. |
| `--modes-disable-force-standard-topk` | Keeps the original top-k output format selection. |

## Environment Variables

The CLI arguments above are the preferred launch surface. The equivalent
`SGLANG_MODES_*` environment variables remain supported for scripts and
debugging.

| Variable | Description | Default |
|---|---|---|
| `SGLANG_ENABLE_MODES` | Enables MoDES route masking. | `0` |
| `SGLANG_MODES_CONFIG` | Loads a reusable MoDES config containing `alpha` and `tau_text`. Supports direct saved configs and `modes_select_config` outputs. | unset |
| `SGLANG_MODES_SAVE_CONFIG` | Saves the effective auto-calibrated config once alpha/tau are ready. | unset |
| `SGLANG_MODES_ALPHA_PATH` | Path to a JSON alpha list or `{"alpha": [...]}` object. | unset |
| `SGLANG_MODES_AUTO_ALPHA` | Auto-estimates alpha when no alpha path is set. | `1` |
| `SGLANG_MODES_ALPHA_MIN` | Lower clamp for auto alpha. | `0.25` |
| `SGLANG_MODES_ALPHA_MAX` | Upper clamp for auto alpha. | `4.0` |
| `SGLANG_MODES_TAU_TEXT` | Fixed text threshold. When unset, auto tau is used. | unset |
| `SGLANG_MODES_AUTO_TAU` | Enables auto tau calibration when no fixed tau is set. | `1` |
| `SGLANG_MODES_TARGET_SKIP_RATE` | Target routed-slot skip rate used by auto tau. | `0.13` |
| `SGLANG_MODES_CALIBRATION_ROUTES` | Number of routed scores collected before auto tau is finalized. | `500000` |
| `SGLANG_MODES_MIN_EXPERTS_PER_TOKEN` | Minimum routed experts kept per token after masking. | `0` |
| `SGLANG_MODES_METRICS_PATH` | Optional JSON metrics output path. | unset |
| `SGLANG_MODES_METRICS_FLUSH_INTERVAL` | Number of MoE calls between metrics writes. | `100` |
| `SGLANG_MODES_FORCE_STANDARD_TOPK` | Forces standard top-k tensors for this patch. | `1` |

Metrics include calibrated alpha, configured and effective tau, calibration
progress, `route_count`, `skip_count`, total `skip_rate`, and post-calibration
`active_skip_rate`. When config reuse is enabled, metrics also include
`config_path`, `config_loaded`, `loaded_alpha`, and `save_config_path`.

## Accuracy-Constrained Selection

The server can auto-calibrate MoDES from router statistics, but it cannot know
task accuracy without labels. To enforce an accuracy constraint such as
`absolute_accuracy_drop <= 0.05`, run a labeled benchmark for the default model
and for several MoDES candidates, then select the best measured candidate:

```bash
python -m sglang.srt.layers.moe.modes_select_config \
  --baseline /path/to/default_result.json \
  --candidate /path/to/modes_skip_10.json \
  --candidate /path/to/modes_skip_13.json \
  --candidate /path/to/modes_skip_20.json \
  --max-accuracy-drop 0.05 \
  --optimize samples_per_s \
  --output /path/to/best_modes_config.json
```

The helper chooses the highest-throughput candidate whose measured accuracy
drop versus baseline is at most five absolute percentage points. It is meant for
offline benchmark selection; the online server path remains one-switch by
default. The helper output can be passed directly to `--modes-config`:

```bash
python -m sglang.launch_server \
  --model-path /models/pubulic-models/Qwen3.5-35B-A3B-FP8 \
  --moe-runner-backend triton \
  --enable-modes \
  --modes-config /path/to/best_modes_config.json
```

## Current Limitations

- Text-only thresholding. Vision/text token-specific thresholds are not wired
  into `ForwardBatch` yet.
- Experimental route masking only. It masks routed contribution by zeroing
  weights. Turning masked routes into actual compute skips requires runner or
  kernel support for dropping routed slots before fused expert execution.
- Auto alpha/tau are skip-rate calibrations, not label-aware accuracy
  optimizers. A new model or dataset should still be validated with
  `modes_select_config` or an equivalent labeled benchmark before claiming
  accuracy and throughput targets.
- Not intended for `triton_kernel` or bypassed fused-top-k backends in this
  first version. Use `--moe-runner-backend triton` for the initial benchmark.
- CUDA graph compatibility has not been validated.
