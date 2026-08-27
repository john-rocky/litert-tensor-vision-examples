// Copyright 2026 The Google AI Edge Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Minimal repro for FINDINGS V3: emitting the `odml.layer_norm` composite
// makes the whole model fail to compile on the GPU delegate, instead of
// falling back to the composite's own decomposition.
//
// One graph, one signature, one op. `--norms=raw` emits the decomposition
// directly (mean / sub / mul / mean / rsqrt / mul / add) and is the control;
// `--norms=composite` wraps that exact decomposition in a StableHLO composite
// named `odml.layer_norm`. The two graphs are numerically identical by
// construction — the composite carries the raw form as its decomposition
// region — so any difference in outcome is delegate-side.
//
// Expected (observed on the SAM2 graph, not yet isolated before this):
//   --norms=raw       --accelerator=cpu  -> OK
//   --norms=raw       --accelerator=gpu  -> OK
//   --norms=composite --accelerator=cpu  -> OK (runtime uses the decomposition)
//   --norms=composite --accelerator=gpu  -> compile fails
//
// Build (drop this directory into LiteRT/tensor/examples/ and build from the
// repo root):
//   bazel build -c opt //tensor/examples/layer_norm_repro:layer_norm_repro
// Run the four cells:
//   for n in raw composite; do for a in cpu gpu; do \
//     ./layer_norm_repro --norms=$n --accelerator=$a; done; done

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"
#include "tensor/tensor.h"

ABSL_FLAG(std::string, norms, "composite",
          "raw|composite — plain ops or the odml.layer_norm composite");
ABSL_FLAG(std::string, accelerator, "gpu", "cpu|gpu");
ABSL_FLAG(std::string, tflite_path, "/tmp/layer_norm_repro.tflite",
          "Where to serialize the one-op model");
ABSL_FLAG(int, tokens, 64, "Token count N in the [1, N, C] input");
ABSL_FLAG(int, channels, 256, "Channel count C (the normalized axis)");
ABSL_FLAG(double, eps, 1e-5, "LayerNorm epsilon");
ABSL_FLAG(std::string, gpu_precision, "default", "default|fp32");
ABSL_FLAG(std::string, gpu_buffer_storage, "buffer",
          "default|buffer|texture2d");
ABSL_FLAG(bool, extra_op, false,
          "Prepend a plain ADD so the graph has a GPU-delegatable op besides "
          "the composite — distinguishes 'the composite breaks the compile' "
          "from 'zero delegatable ops breaks the compile'");

namespace {

using ::litert::tensor::LitertDynamicRunner;
using ::litert::tensor::ModelFactory;
using ::litert::tensor::OwningCpuBuffer;
using ::litert::tensor::StableHLOComposite;
using ::litert::tensor::StableHLOCompositeOptions;
using ::litert::tensor::Type;

using TfTensor = ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>;

TfTensor ConstFloats(const std::vector<float>& values,
                     const std::vector<int>& shape, const std::string& name) {
  return TfTensor({.name = name,
                   .type = Type::kFP32,
                   .shape = shape,
                   .buffer = OwningCpuBuffer::Copy<Type::kFP32>(values)});
}

TfTensor ConstScalar(float value) {
  return TfTensor({.type = Type::kFP32,
                   .shape = {1},
                   .buffer = OwningCpuBuffer::Copy<Type::kFP32>({value})});
}

std::vector<uint8_t> EpsilonAttributes(float eps) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Float("epsilon", eps); });
  fbb.Finish();
  return fbb.GetBuffer();
}

// The decomposition, and the control graph. Last-axis LayerNorm with an
// affine weight and bias — byte-for-byte the form the composite carries.
TfTensor LayerNormRaw(const TfTensor& x, const TfTensor& weight,
                      const TfTensor& bias, float eps) {
  int last = static_cast<int>(x.GetShape().size()) - 1;
  TfTensor mean = Mean(x, {last}, /*keep_dims=*/true);
  TfTensor centered = Sub(x, mean);
  TfTensor var = Mean(Mul(centered, centered), {last}, /*keep_dims=*/true);
  TfTensor normed = Mul(centered, Rsqrt(Add(var, ConstScalar(eps))));
  return Add(Mul(normed, weight), bias);
}

TfTensor LayerNormComposite(const TfTensor& x, const TfTensor& weight,
                            const TfTensor& bias, float eps) {
  StableHLOCompositeOptions opts{.name = "odml.layer_norm",
                                 .composite_attributes =
                                     EpsilonAttributes(eps)};
  return StableHLOComposite(
      opts,
      [eps](TfTensor dx, TfTensor dw, TfTensor db) {
        return LayerNormRaw(dx, dw, db, eps);
      },
      x, weight, bias);
}

absl::Status Run() {
  const std::string norms = absl::GetFlag(FLAGS_norms);
  if (norms != "raw" && norms != "composite") {
    return absl::InvalidArgumentError("--norms must be raw|composite");
  }
  const std::string accel = absl::GetFlag(FLAGS_accelerator);
  if (accel != "cpu" && accel != "gpu") {
    return absl::InvalidArgumentError("--accelerator must be cpu|gpu");
  }
  const int n = absl::GetFlag(FLAGS_tokens);
  const int c = absl::GetFlag(FLAGS_channels);
  const float eps = static_cast<float>(absl::GetFlag(FLAGS_eps));

  // --- Graph: x [1, N, C] -> LayerNorm(last axis) -> y [1, N, C] ---
  TfTensor x({.name = "x", .type = Type::kFP32, .shape = {1, n, c}});
  std::vector<float> gamma(c), beta(c);
  for (int i = 0; i < c; ++i) {
    gamma[i] = 1.0f + 0.01f * static_cast<float>(i % 7);
    beta[i] = 0.001f * static_cast<float>(i % 5);
  }
  TfTensor weight = ConstFloats(gamma, {c}, "gamma");
  TfTensor bias = ConstFloats(beta, {c}, "beta");

  // Optional GPU-delegatable op in front of the norm, so the graph is not
  // "composite only".
  TfTensor in = absl::GetFlag(FLAGS_extra_op) ? Add(x, ConstScalar(0.5f)) : x;

  TfTensor y = (norms == "composite")
                   ? LayerNormComposite(in, weight, bias, eps)
                   : LayerNormRaw(in, weight, bias, eps);
  y.SetName("y");

  ModelFactory factory;
  {
    std::vector<::litert::tensor::TensorHandle> ins{x}, outs{y};
    auto status = factory.AddSignature(ins, outs, "f");
    if (!status.ok()) return status;
  }
  const std::string tflite_path = absl::GetFlag(FLAGS_tflite_path);
  auto save_status = factory.Save(tflite_path);
  if (!save_status.ok()) return save_status;
  std::cout << "norms: " << norms << " | accelerator: " << accel
            << " | shape: [1," << n << "," << c << "]" << std::endl;
  std::cout << "serialized: " << tflite_path << std::endl;

  // --- Compile. This is the step under test. ---
  auto env = ::litert::Environment::Create({});
  if (!env) return absl::InternalError("Environment::Create failed");
  auto options = ::litert::Options::Create();
  if (!options) return absl::InternalError("Options::Create failed");
  const bool use_gpu = accel == "gpu";
  options->SetHardwareAccelerators(use_gpu ? ::litert::HwAccelerators::kGpu
                                           : ::litert::HwAccelerators::kCpu);
  if (use_gpu) {
    auto gpu_options = options->GetGpuOptions();
    if (gpu_options) {
      const std::string precision = absl::GetFlag(FLAGS_gpu_precision);
      if (precision == "fp32") {
        gpu_options->SetPrecision(::litert::GpuOptions::Precision::kFp32);
      } else if (precision != "default") {
        return absl::InvalidArgumentError(
            "--gpu_precision must be default|fp32");
      }
      const std::string storage = absl::GetFlag(FLAGS_gpu_buffer_storage);
      if (storage == "buffer") {
        gpu_options->SetBufferStorageType(
            ::litert::GpuOptions::BufferStorageType::kBuffer);
      } else if (storage == "texture2d") {
        gpu_options->SetBufferStorageType(
            ::litert::GpuOptions::BufferStorageType::kTexture2D);
      } else if (storage != "default") {
        return absl::InvalidArgumentError(
            "--gpu_buffer_storage must be default|buffer|texture2d");
      }
      std::cout << "gpu precision: " << precision << ", storage: " << storage
                << std::endl;
    }
  }

  auto runner_or = LitertDynamicRunner::Create(*env, tflite_path, *options);
  if (!runner_or.ok()) {
    std::cout << "COMPILE FAILED: " << runner_or.status() << std::endl;
    return runner_or.status();
  }
  auto runner = std::move(*runner_or);
  std::cout << "compile: OK" << std::endl;

  // --- Run, so a silent CPU fallback is distinguishable from a real run. ---
  std::vector<float> input(static_cast<size_t>(n) * c);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = std::sin(static_cast<float>(i) * 0.017f) * 3.0f;
  }
  auto st = runner.SetInput(
      "f", "x",
      ::litert::tensor::Create("x", Type::kFP32, {1, n, c}, input));
  if (!st.ok()) return st;
  st = runner.Run("f");
  if (!st.ok()) return st;

  auto t = runner.GetOutput("f", "y");
  if (!t.ok()) return t.status();
  auto buffer = t->GetBuffer();
  if (!buffer.ok()) return buffer.status();
  auto lock = buffer->Lock();
  const float* data = reinterpret_cast<const float*>(lock.data());
  const size_t count = lock.size() / sizeof(float);

  double sum = 0.0, sumsq = 0.0;
  for (size_t i = 0; i < count; ++i) {
    sum += data[i];
    sumsq += static_cast<double>(data[i]) * data[i];
  }
  std::cout << "run: OK | out[0..3] = " << data[0] << ", " << data[1] << ", "
            << data[2] << ", " << data[3] << " | mean "
            << (sum / static_cast<double>(count)) << " | rms "
            << std::sqrt(sumsq / static_cast<double>(count)) << std::endl;
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  auto status = Run();
  if (!status.ok()) {
    std::cerr << "FAILED: " << status << std::endl;
    return 1;
  }
  return 0;
}
