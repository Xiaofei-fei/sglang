# Experimental MoDES Route Masking

This fork includes an experimental text-only MoDES integration for MoE models.
It masks low-importance routed expert slots after top-k routing and before the
MoE runner consumes the route list.

## Launch

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
skip  = score < modes_tau_text
```

Skipped slots are written as:

```text
topk_weight = 0
topk_id = -1
```

`topk_id = -1` follows SGLang's existing padded-route convention. The
integration is only applied to standard top-k outputs. When MoDES is enabled,
SGLang forces standard top-k by default so the mask can be applied outside fused
top-k kernels.

Set `SGLANG_MODES_FORCE_STANDARD_TOPK=0` to disable this behavior.

## Current Limitations

- Text-only thresholding. Vision/text token-specific thresholds are not wired
  into `ForwardBatch` yet.
- Experimental route masking only. It should be benchmarked with the selected
  MoE runner to verify that skipped routes reduce actual expert compute.
- Not intended for `triton_kernel` or bypassed fused-top-k backends in this
  first version. Use `--moe-runner-backend triton` for the initial benchmark.
- CUDA graph compatibility has not been validated.
