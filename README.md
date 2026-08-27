# LiteRT Tensor API vision examples (SAM2.1 hiera-tiny)

The SAM2.1 image path authored directly with the [LiteRT Tensor
API](https://github.com/google-ai-edge/LiteRT/tree/main/tensor) — no
converter. Companion repo to
[litert-tensor-audio-examples](https://github.com/john-rocky/litert-tensor-audio-examples)
(same overlay pattern, same findings-ledger format).

- `sam2_image/` — encoder + prompt encoder + mask decoder in one
  two-signature flatbuffer, NHWC end to end:
  `encode_image` (pixels [1,512,512,3] -> image_embeddings + feat_s1 +
  feat_s0, with conv_s0/s1 and the no-memory embedding folded in) and
  `decode_mask` (those + point_coords [1,1,2] -> 3 multimask logits +
  iou_scores + object_score). The prompt encoder lives in-graph (Sin/Cos
  + a transposed-Gaussian projection; the label logic is constant-folded
  for the static [positive, not_a_point] layout). Baked constants:
  the 128-grid re-composed positional embedding, the dense image PE grid,
  the no-mask dense row, and the output tokens — no gather/broadcast at
  runtime. Windowing is expressed with <=4-D reshapes only; attention is
  rank-4 everywhere. Toggles: `--attention=raw|sdpa|rbmm` (the
  `odml.runtime_bmm` QK+AV pair), `--hypernet=raw|rbmm`,
  `--norms=raw|composite`, `--upsampler=transpose|d2s` (an exact
  4x(1x1)+concat+DepthToSpace expansion of the k2/s2 transposed conv —
  the taps never overlap, so the sums are identical),
  `--split_dir` (single-signature exports for harnesses that only run a
  model's first signature).
- `sam2_image/verify/` — `sam2_torch_ref.py`: parity vs transformers
  `facebook/sam2.1-hiera-tiny` fp32 at 512 on the `--dump_dir` outputs;
  `mlx_512_bench.py`: an MLX baseline on the identical fixture/protocol.
- `sam2_video/` — the SAM2.1 VIDEO tracking path (memory attention +
  memory encoder + object pointers + video mask decoder), authored on the
  Tensor API at the model-native 1024. Five signatures in one flatbuffer
  sharing weights: `encode` (the image-path Hiera encoder reused with the
  no-memory embedding zeroed, so it emits the raw feature map),
  `memcond7`/`memcond2` (memory attention over a fixed bank of 7 or 2
  spatial memory slots + 64 object-pointer tokens; unused slots masked
  additively — numerically identical to the reference's variable-length
  bank), `decode` (video mask decoder: sparse prompt and a `nomem` scalar
  as inputs, all four mask tokens + iou + object pointers + object score
  out; best-mask pick on the host), `memorize` (mask downsampler +
  ConvNeXt fuser -> 64-ch spatial memory, occlusion embedding applied via
  an `occ` input). RoPE is the rotate-half form with the head-dim
  permutation baked into the checkpoint's q/k projections at export time
  and the deinterleaved cos/sin tables baked as constants — no new op
  class anywhere in the video stack. `sam2v_main` carries the whole
  per-frame host loop (bank bookkeeping, pointer temporal encoding,
  mask_for_mem construction), a bench mode and `--dump_dir` parity dumps.
- `sam2_video/verify/` — `export_weights_1024.py` (fp32 export of the
  full video stack from `facebook/sam2.1-hiera-tiny`, HF->checkpoint
  naming, conv layouts pre-permuted to TFLite, RoPE bake with an in-run
  q·k equivalence check); `verify_video_1024.py` (synthetic moving-disk
  clip, HF streaming reference, per-frame compare of mask / object score
  / pointer / memory / memory-attention output); `probe_graphs.py`
  (per-graph isolation: each signature run on inputs captured from the
  HF modules themselves); `mirror_encoder.py` (block-by-block encoder
  parity probe in torch).

## Building

The directory is an overlay for a LiteRT checkout (tested at a19d8fa):

```
cp -r sam2_image sam2_video <litert>/tensor/examples/
cd <litert>
bazel build --config=macos //tensor/examples/sam2_image:sam2_main \
  //tensor/examples/sam2_video:sam2v_main
```

macOS GPU runs need the working directory set to
`<litert>/litert/prebuilt/macos_arm64` so `libLiteRtMetalAccelerator.dylib`
resolves.

```
sam2_main --weights=<sam2_tiny_512.safetensors> \
  --accelerator=gpu --gpu_precision=fp16 --gpu_buffer_storage=buffer \
  --runs=20 --warmup=5 [--dump_dir=<dir>] [--split_dir=<dir>]
```

Video tracking (weights from `sam2_video/verify/export_weights_1024.py`,
clip + reference + compare via `sam2_video/verify/verify_video_1024.py`):

```
python sam2_video/verify/verify_video_1024.py clip     # frames.f32
python sam2_video/verify/verify_video_1024.py ref      # HF reference
sam2v_main --weights=<sam2_tiny_1024_video.safetensors> \
  --frames_file=<frames.f32> --frames=10 --nmm=7 \
  --accelerator=gpu --gpu_precision=fp16 --gpu_buffer_storage=buffer \
  [--dump_dir=<dir>] [--bench_loops=3]
python sam2_video/verify/verify_video_1024.py compare --dump_dir=<dir> --nmm 7
```

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

Three bench windows, each with a sam2_image anchor arm that matched the
month-old stage-1 numbers (8.0 / 1.06 ms); all cells drift < 0.6%
across windows.

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
  blocked on Metal by the second-Run buffer re-registration gap (audio
  ledger #15, re-confirmed on this pin 2026-08-27); the CPU path of that
  contract is proven bit-exact over 64 chained steps on the audio side.

See `FINDINGS.md` for the delegate/runtime observations collected on the
way (the audio repo's ledger format).
