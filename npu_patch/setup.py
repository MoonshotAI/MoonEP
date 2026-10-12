"""
moonep build script.

Two mutually exclusive build paths:

1. CUDA (default on GPU hosts): compiles the CUDA extension `moonep._C`
   (the pybind11 module in csrc/bindings.cu) exactly as before. Uploaded via:

     pip install -e .                              # editable install (preferred)

   or, for a quick in-place build without installing:

     python setup.py build_ext --inplace

   (from the npu_patch/ directory containing this file)

2. Ascend / NPU (when an Ascend toolchain is detected): compiles the
   vendored `npu_patch._C` extension from `npu_patch/native/csrc/` and
   links the AscendC kernel library (`npu_patch/lib/libascend_moonep_kernels.so`)
   built by `npu_patch/scripts/build_base.sh` (wrapper: `build_a5.sh`). The GPU
   logic is unchanged.

The backend is selected at build time by `_is_ascend_build()`: it prefers an
explicit `MOONEP_BUILD_BACKEND`, otherwise it requires an Ascend toolchain
(`ASCEND_HOME_PATH` set and `torch_npu` importable) — never `import torch_npu`
on a plain GPU host.
"""

import os
import platform
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension


# setup.py now lives in npu_patch/: the repo root (home of ``moonep``/``csrc``)
# is one level up, and find_packages must scan from there so both the GPU
# (moonep) and NPU (npu_patch) packages are installed by `pip install -e .`.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = Path(REPO_ROOT)


def _build_backend() -> str:
    """'npu' when an Ascend toolchain is present, else 'cuda'."""
    env = os.environ.get("MOONEP_BUILD_BACKEND", "").strip().lower()
    if env:
        if env not in ("cuda", "npu"):
            raise RuntimeError(f"MOONEP_BUILD_BACKEND must be cuda or npu, got {env!r}")
        return env
    if os.environ.get("ASCEND_HOME_PATH"):
        try:
            import torch_npu
            return "npu"
        except Exception:
            return "cuda"
    return "cuda"


def host_arch_dir():
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        return "aarch64-linux"
    if machine in {"x86_64", "amd64"}:
        return "x86_64-linux"
    return f"{machine}-linux"


def existing_paths(paths):
    return [str(path) for path in paths if path.exists()]


def _npu_extension():
    """CppExtension for the vendored Ascend `npu_patch._C` backend."""
    from torch.utils.cpp_extension import CppExtension

    import torch_npu

    asc_home = Path(os.environ["ASCEND_HOME_PATH"])
    asc_arch = asc_home / host_arch_dir()
    shmem_home = Path(
        os.environ.get("SHMEM_HOME_PATH",
                       ROOT / "npu_patch" / "3rdparty" / "shmem" / "install")
    )
    kernel_lib_dir = Path(
        os.environ.get("ASCEND_MOONEP_KERNEL_LIB_DIR", ROOT / "npu_patch" / "lib")
    )
    if not (kernel_lib_dir / "libascend_moonep_kernels.so").exists():
        raise RuntimeError(
            f"Kernel library not found in {kernel_lib_dir}. Run "
            "npu_patch/scripts/build_base.sh -soc_type <SOC_TYPE> first."
        )
    torch_npu_root = Path(torch_npu.__file__).resolve().parent
    native = ROOT / "npu_patch" / "native"

    include_dirs = (
        [REPO_ROOT, native]
        + [str(native / "include"), str(native / "csrc")]
        + existing_paths(
            [
                asc_home / "include",
                shmem_home / "shmem" / "include",
                shmem_home / "shmem" / "src",
                shmem_home / "memfabric_hybrid" / "include" / "smem" / "host",
                torch_npu_root / "include",
                torch_npu_root.parent,
                asc_arch / "include",
                asc_arch / "pkg_inc" / "runtime",
                asc_arch / "runtime" / "include",
                asc_arch / "include" / "runtime",
            ]
        )
    )

    library_dirs = existing_paths(
        [
            kernel_lib_dir,
            asc_home / "lib64",
            shmem_home / "shmem" / "lib",
            shmem_home / "memfabric_hybrid" / "lib",
            torch_npu_root / "lib",
            asc_arch / "lib64",
        ]
    )

    rpath_dirs = existing_paths(
        [
            kernel_lib_dir,
            asc_home / "lib64",
            shmem_home / "shmem" / "lib",
            shmem_home / "memfabric_hybrid" / "lib",
            asc_arch / "lib64",
        ]
    )

    # Sources must be relative to the setup.py directory (npu_patch/) —
    # setuptools rejects absolute paths. Native sources live under
    # `npu_patch/native/csrc`, i.e. `native/csrc` from here.
    sources = [
        os.path.join("native", "csrc", "pybind.cpp"),
        os.path.join("native", "csrc", "acl_shared_buffer.cpp"),
        os.path.join("native", "csrc", "torch_classes.cpp"),
        os.path.join("native", "csrc", "launcher.cpp"),
        os.path.join("native", "csrc", "ops", "moonep_combine.cpp"),
        os.path.join("native", "csrc", "ops", "moonep_combine_udma.cpp"),
        os.path.join("native", "csrc", "ops", "moonep_dispatch.cpp"),
        os.path.join("native", "csrc", "ops", "planning.cpp"),
        os.path.join("native", "csrc", "ops", "prefetch.cpp"),
        os.path.join("native", "csrc", "ops", "grad_reduce.cpp"),
        os.path.join("native", "csrc", "shmem_instance.cpp"),
    ]
    return CppExtension(
        name="npu_patch._C",
        sources=sources,
        define_macros=[],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["ascend_moonep_kernels", "torch_npu", "ascendcl", "runtime", "shmem", "dl"],
        extra_compile_args=["-O2", "-std=c++17"],
        extra_link_args=["-Wl,-rpath,$ORIGIN/lib"]
        + [f"-Wl,-rpath,{path}" for path in rpath_dirs],
    )


BACKEND = _build_backend()

if BACKEND == "npu":
    ext_modules = [_npu_extension()]
    cmdclass = {"build_ext": BuildExtension}
else:
    # --- CUDA path: unchanged from upstream MoonEP ---
    from torch.utils.cpp_extension import CUDAExtension

    nvcc_flags = [
        "-O3",
        "--use_fast_math",
        "-std=c++20",
        "--expt-extended-lambda",
        "--expt-relaxed-constexpr",
        "-forward-unknown-to-host-compiler",
        "-Xcompiler=-Wno-psabi",
        "-Xcompiler=-fno-strict-aliasing",
        "-DNDEBUG",
        "-lineinfo",
        "-diag-suppress=3189",
        "-ftemplate-backtrace-limit=0",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]

    cxx_flags = [
        "-O3",
        "-fno-strict-aliasing",
        "-Wno-psabi",
    ]

    ext_modules = [
        CUDAExtension(
            name="moonep._C",
            sources=[
                os.path.join("..", "csrc", "bindings.cu"),
            ],
            include_dirs=[
                os.path.join(REPO_ROOT, "csrc"),
            ],
            libraries=["cuda"],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        ),
    ]
    cmdclass = {"build_ext": BuildExtension}


_PACKAGES = find_packages(
    where=REPO_ROOT, include=["moonep", "moonep.*", "npu_patch", "npu_patch.*"])


def _package_data():
    """Bundle expected runtime shared libraries for the vendored backend."""
    if BACKEND != "npu":
        return {}
    # Globs are relative to the `npu_patch` package directory.
    libs = ["lib/*.so", "lib/*.so.*"]
    return {"npu_patch": libs}


setup(
    name="moonep",
    version="0.0.1",
    packages=_PACKAGES,
    package_dir={"": ".."},
    package_data=_package_data(),
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
