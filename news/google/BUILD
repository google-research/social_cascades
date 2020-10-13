load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary")

licenses(["notice"])

pytype_strict_binary(
    name = "launch_optimizer_main",
    srcs = ["launch_optimizer_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/brain/frameworks/xcloud",
        "//learning/deepmind/xmanager:hyper",
        "//learning/vizier/service:automated_stopping_py_pb2",
        "//learning/vizier/service:vizier_py_pb2",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/termcolor",
    ],
)

pytype_strict_binary(
    name = "launch_optimizer_gcnrnn",
    srcs = ["launch_optimizer_gcnrnn.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/brain/frameworks/xcloud",
        "//learning/deepmind/xmanager:hyper",
        "//learning/vizier/service:automated_stopping_py_pb2",
        "//learning/vizier/service:vizier_py_pb2",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/termcolor",
    ],
)
