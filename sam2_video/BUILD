# SAM2.1 hiera-tiny VIDEO tracking path on the C++ Tensor API. Drop this
# directory into LiteRT/tensor/examples/sam2_video/ (next to sam2_image,
# which it reuses for the encoder and image-path infrastructure) and build
# from the repo root.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//litert/build_common:litert_build_defs.bzl", "commandline_flag_copts")
load(
    "//litert/build_common:special_rule.bzl",
    "litert_android_linkopts",
    "litert_gpu_accelerator_prebuilts",
)

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "sam2v_graph",
    srcs = ["sam2v_graph.cc"],
    hdrs = ["sam2v_graph.h"],
    deps = [
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/tflite:arithmetic_tflite",
        "//tensor/examples/sam2_image:sam2_config",
        "//tensor/examples/sam2_image:sam2_graph",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "sam2v_weights",
    srcs = ["sam2v_weights.cc"],
    hdrs = ["sam2v_weights.h"],
    deps = [
        ":sam2v_graph",
        "//tensor",
        "//tensor/examples/gemma3:safetensor_loader",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "sam2v_main",
    srcs = ["sam2v_main.cc"],
    copts = select({
        "//litert:android": [],
        "//conditions:default": ["-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1"],
    }) + commandline_flag_copts(),
    data = litert_gpu_accelerator_prebuilts(),
    linkopts = litert_android_linkopts(),
    deps = [
        ":sam2v_graph",
        ":sam2v_weights",
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
        "//tensor/examples/sam2_image:sam2_config",
        "//tensor/examples/sam2_image:sam2_graph",
        "//tensor/examples/sam2_image:sam2_weights",
        "//tensor/runners/litert:litert_dynamic_runner",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)
