"""Regression tests for errors crossing the CUDA extension boundary."""

import subprocess
import sys
import textwrap


def test_invalid_vmm_handle_raises_without_terminating_process():
    """A driver error must be catchable instead of calling exit()."""
    script = textwrap.dedent(
        """
        from moonep._C import nvl_release_mem_handle

        try:
            nvl_release_mem_handle(-1)
        except RuntimeError as error:
            assert "CUDA driver call `cuMemRelease" in str(error), str(error)
        else:
            raise AssertionError("invalid CUDA allocation handle was accepted")

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
