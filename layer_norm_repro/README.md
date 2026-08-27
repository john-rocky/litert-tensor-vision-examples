# `odml.layer_norm` composite — minimal repro (FINDINGS V3)

One graph, one signature, one op. Isolates the V3 ledger entry, which until
now could only be reproduced by running the whole SAM2 image path
(`sam2_main --norms=composite --accelerator=gpu`).

## What it does

`--norms=raw` builds the LayerNorm decomposition directly:

```
mean(x, last) -> sub -> mul(centered, centered) -> mean(last) -> add(eps)
              -> rsqrt -> mul -> mul(gamma) -> add(beta)
```

`--norms=composite` wraps **that exact function** in a StableHLO composite
named `odml.layer_norm`, carrying `epsilon` as a flexbuffer attribute and the
raw form as its decomposition region. The two graphs are numerically identical
by construction, so any difference in outcome is delegate-side, not a modelling
difference.

## Measured result

macOS arm64 (M4 Max), LiteRT at local pin `a19d8fa8` (2026-07-16) + the #8796
cherry-pick, prebuilt Metal accelerator, 2026-07-27.

| `--norms` | `--accelerator` | `--extra_op` | result |
|---|---|---|---|
| raw | cpu | — | compile OK, run OK — `out[0]=-0.519143`, rms 1.02971 |
| raw | gpu | — | compile OK, run OK — `out[0]=-0.519043`, rms 1.02957 |
| composite | cpu | — | compile OK, run OK — **bit-identical to raw/cpu** |
| composite | gpu | — | **COMPILE FAILED** |
| raw | gpu | yes | compile OK, run OK |
| composite | gpu | yes | **COMPILE FAILED** |

The GPU rows differ from the CPU rows in the last digits (fp16 execution), so
those are genuine GPU runs and not a silent CPU fallback.

The delegate message, with `--extra_op` so the graph is not composite-only:

```
ERROR: Following operations are not supported by GPU delegate:
STABLEHLO_COMPOSITE: odml.layer_norm
1 operations will run on the GPU, and the remaining 1 operations will run on the CPU.
COMPILE FAILED: INTERNAL: ... Failed to compile model
```

**The delegate identifies the composite as unsupported, computes a valid
partition (1 op GPU / 1 op CPU), and then the compile fails anyway.** Without
`--extra_op` the same happens with the partition reported as "No operations
will run on the GPU, and all 1 operations will run on the CPU." So the failure
is not "nothing left to delegate" — the `raw + --extra_op` control on the same
backend compiles and runs.

This sharpens the original ledger wording ("fails the whole model compile
instead of falling back"): the fallback *is* planned and announced, and the
compile still fails.

## Running it

```
bazel build -c opt //tensor/examples/layer_norm_repro:layer_norm_repro
D=bazel-bin/tensor/examples/layer_norm_repro
./$D/layer_norm_repro --norms=composite --accelerator=gpu
```

On macOS the `data = litert_gpu_accelerator_prebuilts()` select did not stage
the accelerator into runfiles in this checkout, and the GPU arm then reports
`gpu_registry.cc:131 GPU accelerator could not be loaded and registered` for
*every* graph including the raw control. Stage it next to the binary first:

```
cp <output_base>/external/litert_prebuilts/macos_arm64/libLiteRtMetalAccelerator.dylib $D/
```

Always run the `--norms=raw --accelerator=gpu` control before trusting a
composite failure — that is what caught this the first time.

## Flags

| flag | default | note |
|---|---|---|
| `--norms` | `composite` | `raw` is the control |
| `--accelerator` | `gpu` | `cpu` is the control |
| `--tokens` / `--channels` | 64 / 256 | the `[1, N, C]` input; vary to check shape-independence |
| `--eps` | 1e-5 | the composite attribute |
| `--gpu_precision` | `default` | `fp32` to rule out precision |
| `--gpu_buffer_storage` | `buffer` | `texture2d` reproduces the storage-mode difference documented in the audio ledger |
| `--tflite_path` | `/tmp/layer_norm_repro.tflite` | the serialized one-op model, for inspection |

## Context

The registry that decides whether a composite is parsed is visible in the OSS
tree at `ml_drift_delegate/delegate/composite/` — the factory in
`custom_parsers.cc` and the map in `ir/custom_parsers.cc`. As of 2026-07-27
both list exactly three composites (`odml.cache_update`, `odml.runtime_bmm`,
`moe`) and `odml.layer_norm` is in neither, which is consistent with what this
repro shows. Note that `odml.scaled_dot_product_attention`, `rms_norm` and
`group_norm` are not in those registries either yet demonstrably run, so the
visible registry is not the whole picture.

Building anything under `ml_drift_delegate/` from the OSS tree is currently not
possible: 280 files there reference `@ml_drift`, and `bazel query` reports
`The repository '@@ml_drift' could not be resolved`.

Status: built and run on 2026-07-27; the table above is measured, not
predicted.
