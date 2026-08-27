# Minimal odml.layer_norm composite repro (FINDINGS V3).
# Drop this directory into LiteRT/tensor/examples/layer_norm_repro/ and build
# from the repo root.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("//litert/build_common:litert_build_defs.bzl", "commandline_flag_copts")
load(
    "//litert/build_common:special_rule.bzl",
    "litert_android_linkopts",
    "litert_gpu_accelerator_prebuilts",
)

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "layer_norm_repro",
    srcs = ["layer_norm_repro.cc"],
    copts = select({
        "//litert:android": [],
        "//conditions:default": ["-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1"],
    }) + commandline_flag_copts(),
    data = litert_gpu_accelerator_prebuilts(),
    linkopts = litert_android_linkopts(),
    deps = [
        "//litert/cc:litert_api_with_dynamic_runtime",
        "//litert/cc:litert_common",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_options",
        "//litert/cc/options:litert_gpu_options",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/tflite:arithmetic_tflite",
        "//tensor/backends/tflite:tflite_flatbuffer_conversion",
        "//tensor/runners/litert:litert_dynamic_runner",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)
