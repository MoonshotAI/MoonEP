"""Regression tests for errors crossing the CUDA extension boundary."""

import subprocess
import sys
import textwrap


def test_non_cuda_posix_fd_raises_without_terminating_process():
    """A driver error must be catchable instead of calling exit()."""
    script = textwrap.dedent(
        """
        import os

        from moonep._C import nvl_multicast_import

        fd = os.open(os.devnull, os.O_RDONLY)
        try:
            try:
                nvl_multicast_import(fd)
            except RuntimeError as error:
                assert "cuMemImportFromShareableHandle" in str(error), str(error)
            else:
                raise AssertionError("non-CUDA POSIX fd was accepted")
        finally:
            os.close(fd)

        print("process continued after CUDA driver error")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "process continued after CUDA driver error" in result.stdout
