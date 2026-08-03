"""
moonep build script. Compiles the CUDA extension `moonep._C` (the pybind11
module living in csrc/bindings.cu) into the `moonep` package so it can be
imported as `from moonep import _C` after either:

  pip install -e .                              # editable install (preferred)

or, for a quick in-place build without installing:

  python setup.py build_ext --inplace

The .so lands at `moonep/_C.<abi-tag>.so`.
"""

import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


REPO_ROOT = os.path.abspath(os.path.dirname(__file__))


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
            os.path.join("csrc", "bindings.cu"),
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


setup(
    name="moonep",
    version="0.0.1",
    packages=find_packages(include=["moonep", "moonep.*"]),
    install_requires=[
        "nvidia-cutlass-dsl==4.4.2",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
