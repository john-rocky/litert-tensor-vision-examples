# Findings ledger — LiteRT / delegates, from the SAM2 vision examples

Same format as the audio repo's ledger. One line per finding; status:
`open` = reported, awaiting guidance; `workaround` = we ship around it;
`documented` = behavior note, no action needed. Routing (in-thread vs
GitHub issue) is always the LiteRT team's call — repros stay ready either
way. Backend "Metal" = macOS/iOS prebuilt accelerator at LiteRT a19d8fa;
"CL" = ML Drift / LITERT_CL on a Pixel 8a.

| # | finding | backend | status | repro / notes |
|---|---|---|---|---|
| V1 | A graph Reshape whose input is a constant ("Expected 1 runtime input tensor(s), but node has 0") is rejected by the delegate and the whole model then FAILS to compile — no per-node CPU fallback | Metal | workaround (bake constants at their target shape at build time; no graph Reshape of weights) | hit on no_mem_embed / prompt-token rows; fix in `sam2_graph.cc` |
| V2 | odml.scaled_dot_product_attention: fp32 output exact (CPU to 6 decimals), but at fp16 the mask decoder's two-way shapes compute wrong results (masks corr 0.66 vs CPU, the iou head collapses 0.986 -> 0.0001) while the encoder's window/global shapes stay at 0.9998; raw-op attention at fp16 is fine everywhere | Metal | workaround (raw attention default; sdpa viable for the encoder or at fp32) | `sam2_main --attention=sdpa --gpu_precision=fp16` vs `raw`; per-graph attribution via `--dump_dir` corr |
| V3 | odml.layer_norm is not in the Metal delegate's parsed composite set (rms_norm / group_norm / sdpa / cache_update are) and emitting it fails the whole model compile instead of falling back to the composite's decomposition | Metal | open | `sam2_main --norms=composite --accelerator=gpu`; CPU decomposition matches raw exactly, so the emission is sound |
| V4 | sdpa composite on CPU: XNNPACK substitutes its own lowering for the authored decomposition and the result differs from raw ops on these MHA shapes — the audio repo's #14, reproduced on a vision graph | CPU | workaround (CPU-parity runs use `--attention=raw`) | `sam2_main --attention=sdpa` on CPU vs `raw` |
| V5 | TRANSPOSE_CONV (k2/s2, bias) delegates AND computes correctly on the current CL stack (276/276 nodes, masks corr 0.999983 vs host) — the rejection we hit during the earlier converted-path work does not reproduce; the exact 4x(1x1)+concat+DepthToSpace expansion is kept as a verified fallback (`--upsampler=d2s`, bit-identical on CPU, same speed on Metal) | CL (Pixel 8a) | documented | both decoder variants on-device via `--split_dir` exports + a residency harness |
| V6 | ai-edge-quantizer (float_casting recipe) requires every tensor to be named and crashes on ModelFactory output (intermediate tensors are unnamed): `tensor.name.decode` on None | tooling | open observation | quantize any `sam2_main --tflite_path` output; workaround = ship fp32 weights (the Metal delegate's fp16 execution mode is independent of storage) |
| V7 | MLX `mx.save_safetensors` writes `__metadata__: null`, which the LiteRT-tree safetensors parser rejects ("`__metadata__` value must be JSON object") | tooling | documented (one-line python re-serialization, README) | |

Adopted from the LiteRT side: the <=4-D windowing / rank-4-attention /
baked-constant graph forms carried over from the converted-path work
(litert-samples interactive-segmentation sample, decoder v2) — the rank-4
form is what makes the authored decoder correct first-try on the device
that exposed the rank-3 miscompute.
