# LiteRT Tensor API vision examples (SAM2.1 hiera-tiny)

This repo re-creates SAM2.1, Meta's image and video segmentation model, by hand in C++ with the [LiteRT Tensor API](https://github.com/google-ai-edge/LiteRT/tree/main/tensor), which lets you write a LiteRT graph as code instead of converting a model. `sam2_image/` segments images at 512×512 pixels, and `sam2_video/` tracks an object through video at 1024×1024.

It is my own example code for LiteRT C++ developers, not part of LiteRT. There is no app: you copy it into a LiteRT checkout and build it with Bazel.

| Device, fp16 | Image encoder | Mask decoder |
|---|---|---|
| Apple M4 Max, Metal | 8.0 ms | 1.06 ms |
| iPhone 17 Pro | 36.2 ms | 1.8 ms |

Video tracking takes 94 ms per frame with a memory bank of 7 slots and 64 ms with 2, on the M4 Max in fp16. The image masks match Hugging Face transformers in fp32 (correlation 1.000000). In fp32, the video masks match its streaming reference over a 10-frame synthetic clip (minimum IoU 1.0000), and in fp16 on Metal the minimum video mask IoU is 0.9950. The iPhone numbers come from a test app that is not in this repo.

## Building

Copy both folders into a LiteRT checkout (tested at a19d8fa with [PR #8796](https://github.com/google-ai-edge/LiteRT/pull/8796) cherry-picked, which this code does not use):

```
cp -r sam2_image sam2_video <litert>/tensor/examples/
cd <litert>
bazel build --config=macos //tensor/examples/sam2_image:sam2_main \
  //tensor/examples/sam2_video:sam2v_main
```

## Weights

The weights are not in this repo. Download `mlx/sam2_tiny_512.safetensors` (the 512×512 set, with the positional embedding re-composed for that size) from [mlboydaisuke/SAM2-hiera-tiny-LiteRT](https://huggingface.co/mlboydaisuke/SAM2-hiera-tiny-LiteRT).

MLX writes `__metadata__: null`, which LiteRT's safetensors parser rejects, so save a clean copy once:

```
python -c "from safetensors.numpy import load_file, save_file; \
  save_file(load_file('sam2_tiny_512.safetensors'), 'sam2_tiny_512_clean.safetensors')"
```

## Running

On macOS, the GPU runs need `<litert>/litert/prebuilt/macos_arm64` as the working directory, so that `libLiteRtMetalAccelerator.dylib` resolves. Bazel writes the binaries under `<litert>/bazel-bin/tensor/examples/`, so call them by their full paths.

To time the image model and get a mask:

```
sam2_main --weights=<sam2_tiny_512_clean.safetensors> \
  --accelerator=gpu --gpu_precision=fp16 --gpu_buffer_storage=buffer \
  --runs=20 --warmup=5 [--dump_dir=<dir>] [--split_dir=<dir>]
```

With no input it uses a built-in test image and a point at the center. `--input_file` takes your own image as a raw fp32 file, already normalized, in NHWC [1,512,512,3], and `--point_x` and `--point_y` set the point. `--dump_dir` writes the mask and the other tensors as raw files.

Video tracking needs its own weights: `sam2_video/verify/export_weights_1024.py` exports them from `facebook/sam2.1-hiera-tiny`. `verify_video_1024.py` writes a synthetic clip of a white disk moving on black, makes a Hugging Face reference and compares the result. `--nmm` is the number of memory slots.

```
python sam2_video/verify/verify_video_1024.py clip     # frames.f32
python sam2_video/verify/verify_video_1024.py ref      # HF reference
sam2v_main --weights=<sam2_tiny_1024_video.safetensors> \
  --frames_file=<frames.f32> --frames=10 --nmm=7 \
  --accelerator=gpu --gpu_precision=fp16 --gpu_buffer_storage=buffer \
  [--dump_dir=<dir>] [--bench_loops=3]
python sam2_video/verify/verify_video_1024.py compare --dump_dir=<dir> --nmm 7
```

## Documentation

- [Folders and options](docs/what-is-here.md)
- [Timings and output comparisons](docs/results.md)
- [What I found in the delegates and the runtime](FINDINGS.md)
- [litert-tensor-audio-examples](https://github.com/john-rocky/litert-tensor-audio-examples): the same approach for audio models
