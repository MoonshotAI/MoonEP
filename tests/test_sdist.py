"""CPU-only source archive checks: python -m unittest tests.test_sdist."""

from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


class SourceDistributionTest(unittest.TestCase):
    def test_cuda_header_is_in_source_archive(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            shutil.copytree(
                root,
                source,
                ignore=shutil.ignore_patterns(
                    ".git", "*.egg-info", "__pycache__", "build", "dist"
                ),
            )
            # sdist only needs extension metadata, not a CUDA compiler. Keep
            # the real setup.py and setuptools sdist implementation, replacing
            # only PyTorch's CUDA-dependent extension factory/build command.
            script = """
import runpy
import sys
import types
from setuptools import Extension
from setuptools.command.build_ext import build_ext

for name in ("torch", "torch.utils", "torch.utils.cpp_extension"):
    sys.modules[name] = types.ModuleType(name)
ext = sys.modules["torch.utils.cpp_extension"]
ext.CUDAExtension = Extension
ext.BuildExtension = build_ext
sys.argv = ["setup.py", "sdist"]
runpy.run_path("setup.py", run_name="__main__")
"""
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=source,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            archives = list((source / "dist").glob("*.tar.gz"))
            self.assertEqual(len(archives), 1)
            with tarfile.open(archives[0]) as archive:
                members = {name.split("/", 1)[-1] for name in archive.getnames()}
                self.assertIn("csrc/bindings.cu", members)
                self.assertIn("csrc/nvl_shared_buffer.cuh", members)


if __name__ == "__main__":
    unittest.main()
