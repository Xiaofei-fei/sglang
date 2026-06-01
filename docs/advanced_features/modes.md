# Experimental MoDES Route Masking

This fork includes an experimental text-only MoDES integration for MoE models.
It masks low-importance routed expert slots after top-k routing and before the
MoE runner consumes the route list.

## Launch

By default, MoDES auto-calibrates `tau_text` from the live routed-score
distribution. You only need to provide an alpha file and a target skip rate:

```bash
SGLANG_ENABLE_MODES=1 \
SGLANG_MODES_ALPHA_PATH=/path/to/modes_alpha.json \
SGLANG_MODES_TARGET_SKIP_RATE=0.13 \
SGLANG_MODES_CALIBRATION_ROUTES=500000 \
SGLANG_MODES_MIN_EXPERTS_PER_TOKEN=0 \
SGLANG_MODES_METRICS_PATH=/tmp/modes_metrics.json \
python -m sglang.launch_server \
  --model-path /models/pubulic-models/Qwen3.5-35B-A3B-FP8 \
  --moe-runner-backend triton
```

If `SGLANG_MODES_TAU_TEXT` is not set, the first
`SGLANG_MODES_CALIBRATION_ROUTES` routed scores are used as a calibration
window. SGLang then chooses:

```text
tau_text = quantile(scores, SGLANG_MODES_TARGET_SKIP_RATE)
```

and starts masking routes after the calibration window. For stable benchmark
numbers, run a short warmup first and exclude calibration requests from the
reported latency/throughput.

For fully fixed experiments, set `SGLANG_MODES_TAU_TEXT` explicitly:

```bash
SGLANG_ENABLE_MODES=1 \
SGLANG_MODES_ALPHA_PATH=/path/to/modes_alpha.json \
SGLANG_MODES_TAU_TEXT=0.000438 \
SGLANG_MODES_MIN_EXPERTS_PER_TOKEN=0 \
python -m sglang.launch_server \
  --model-path /models/pubulic-models/Qwen3.5-35B-A3B-FP8 \
  --moe-runner-backend triton
```

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

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `SGLANG_ENABLE_MODES` | Enables MoDES route masking. | `0` |
| `SGLANG_MODES_ALPHA_PATH` | Path to a JSON alpha list or `{"alpha": [...]}` object. | unset |
| `SGLANG_MODES_TAU_TEXT` | Fixed text threshold. When unset, auto tau is used. | unset |
| `SGLANG_MODES_AUTO_TAU` | Enables auto tau calibration when no fixed tau is set. | `1` |
| `SGLANG_MODES_TARGET_SKIP_RATE` | Target routed-slot skip rate used by auto tau. | `0.13` |
| `SGLANG_MODES_CALIBRATION_ROUTES` | Number of routed scores collected before auto tau is finalized. | `500000` |
| `SGLANG_MODES_MIN_EXPERTS_PER_TOKEN` | Minimum routed experts kept per token after masking. | `0` |
| `SGLANG_MODES_METRICS_PATH` | Optional JSON metrics output path. | unset |
| `SGLANG_MODES_METRICS_FLUSH_INTERVAL` | Number of MoE calls between metrics writes. | `100` |
| `SGLANG_MODES_FORCE_STANDARD_TOPK` | Forces standard top-k tensors for this patch. | `1` |

Metrics include the configured and effective tau, calibration progress,
`route_count`, `skip_count`, total `skip_rate`, and post-calibration
`active_skip_rate`.

## Current Limitations

- Text-only thresholding. Vision/text token-specific thresholds are not wired
  into `ForwardBatch` yet.
- Experimental route masking only. It masks routed contribution by zeroing
  weights. Turning masked routes into actual compute skips requires runner or
  kernel support for dropping routed slots before fused expert execution.
- Auto tau is a skip-rate calibration, not a label-aware accuracy optimizer.
  A new model or dataset should still be validated on a labeled benchmark before
  claiming accuracy and throughput targets.
- Not intended for `triton_kernel` or bypassed fused-top-k backends in this
  first version. Use `--moe-runner-backend triton` for the initial benchmark.
- CUDA graph compatibility has not been validated.
