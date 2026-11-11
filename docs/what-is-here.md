# What is here

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
- `layer_norm_repro/`: one graph and one op, to reproduce finding V3 in
  [FINDINGS.md](../FINDINGS.md) without running the whole SAM2 image path.
