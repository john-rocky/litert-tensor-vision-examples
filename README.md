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
  rank-4 everywhere. Toggles: `--attention=raw|sdpa`,
  `--norms=raw|composite`, `--upsampler=transpose|d2s` (an exact
  4x(1x1)+concat+DepthToSpace expansion of the k2/s2 transposed conv —
  the taps never overlap, so the sums are identical),
  `--split_dir` (single-signature exports for harnesses that only run a
  model's first signature).
- `sam2_image/verify/` — `sam2_torch_ref.py`: parity vs transformers
  `facebook/sam2.1-hiera-tiny` fp32 at 512 on the `--dump_dir` outputs;
  `mlx_512_bench.py`: an MLX baseline on the identical fixture/protocol.

## Building

The directory is an overlay for a LiteRT checkout (tested at a19d8fa):

```
cp -r sam2_image <litert>/tensor/examples/
cd <litert>
bazel build --config=macos //tensor/examples/sam2_image:sam2_main
```

macOS GPU runs need the working directory set to
`<litert>/litert/prebuilt/macos_arm64` so `libLiteRtMetalAccelerator.dylib`
resolves.

```
sam2_main --weights=<sam2_tiny_512.safetensors> \
  --accelerator=gpu --gpu_precision=fp16 --gpu_buffer_storage=buffer \
  --runs=20 --warmup=5 [--dump_dir=<dir>] [--split_dir=<dir>]
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

See `FINDINGS.md` for the delegate/runtime observations collected on the
way (the audio repo's ledger format).
