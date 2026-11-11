# Results

## Measured highlights (512, warm medians, two separated windows)

- Parity: masks corr 1.000000, iou_scores within 0.0016, identical
  best-mask pick vs transformers fp32 (`verify/sam2_torch_ref.py`).
- M4 Max Metal, fully delegated: fp16 8.0 ms encoder / 1.06 ms decoder
  (fp32 9.3 / 1.17 = CPU-exact); same-machine MLX python baseline
  9.0 / 1.36.
- iPhone 17 Pro (both apps Release, fp16, two windows each): Tensor API
  36.2 / 1.8 both windows vs mlx-swift 33.7-36.5 / 3.3 — encoder is a
  statistical tie, decoder -45%.
- Pixel 8a (ML Drift / LITERT_CL): encoder 595/595 and decoder 286/286
  nodes delegated, correct on device (masks corr 0.99998 vs host), with
  either upsampler form.

## Measured highlights (video path, 1024, warm medians)

Three bench windows. In each, the image model was timed again and matched
the earlier 8.0 / 1.06 ms. The per-frame totals moved by less than 0.6% between
windows.

- Parity vs the HF streaming reference (`Sam2VideoModel`, fp32, 10-frame
  synthetic clip, chained state): CPU fp32 and Metal fp32 both at
  **min mask-IoU 1.0000** (identical foreground counts every frame,
  max|d| on mask logits 0.008) for BOTH bank sizes (7-slot and 2-slot);
  Metal fp16 min mask-IoU 0.9950 over the chained loop, both bank sizes.
- M4 Max Metal fp16 (buffer storage, all five signatures fully
  delegated), per tracked frame: encode 33.7 ms + memory attention
  53.5 ms (7-slot) / 23.2 ms (2-slot) + decode 1.7 ms + memory encoder
  1.4 ms -> **94 ms/frame (7-slot), 64 ms/frame (2-slot)** end to end
  including the host loop. fp32: 117 / 78 ms/frame. CPU fp32:
  1505 ms/frame (7-slot).
- The memory bank lives host-side as per-frame signature inputs — the
  same contract as the reference pipeline. The in-graph signature-state
  variant (odml.cache_update + the PR #8796 feedback-loop runner) is
  blocked on Metal by the second-Run buffer re-registration gap (finding
  15 in the audio repo's FINDINGS.md, confirmed again at LiteRT a19d8fa
  on 2026-08-27); the CPU path of that
  contract is proven bit-exact over 64 chained steps on the audio side.

See `FINDINGS.md` for the delegate/runtime observations collected on the
way (the audio repo's ledger format).

## Weights

Not included. The 512-baked fp16 set (MLX naming, `pos_embed_full`
re-composed at the 128x128 token grid — slicing the 1024 grid scores corr
0.947 and resizing 0.748 vs PyTorch; re-composing restores 0.9999) is
`mlx/sam2_tiny_512.safetensors` at
[mlboydaisuke/SAM2-hiera-tiny-LiteRT](https://huggingface.co/mlboydaisuke/SAM2-hiera-tiny-LiteRT).
MLX writes `__metadata__: null`, which the LiteRT-tree safetensors parser
rejects — re-serialize once:

```
python -c "from safetensors.numpy import load_file, save_file; \
  save_file(load_file('sam2_tiny_512.safetensors'), 'sam2_tiny_512_clean.safetensors')"
```
