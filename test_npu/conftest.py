# SPDX-License-Identifier: Apache-2.0
"""Pytest fixtures for the ascend-moonep distributed tests.

Ascend port of ``MoonEP/tests/conftest.py``: the ``dist_env`` session fixture
initializes the process group (HCCL) and the ``cleanup_moonep_buffers``
autouse fixture destroys every ``Buffer`` created through
``kernel_test_utils.init_case``. Backend is ``hccl`` and the device is selected
from ``LOCAL_RANK``.
"""

import os
import sys

# Make the shared test helpers (kernel_test_utils / generate_topk_routing /
# planing_reference) and sibling test modules importable both under pytest and
# when a test file is executed directly as a script (e.g. `python
# test_npu/urma/test_moonep_dispatch_udma_with_plan.py`). The helpers live in this
# directory; the per-operator test suites live in the buffer_test/urma/mte
# subpackages and cross-import siblings within the same directory.
_TEST_ROOT = os.path.dirname(os.path.abspath(__file__))
for _d in ("", "buffer_test", "urma", "mte"):
    _p = os.path.join(_TEST_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest
import torch
import torch.distributed as dist

import torch_npu


@pytest.fixture(autouse=True)
def cleanup_moonep_buffers():
    yield
    from kernel_test_utils import destroy_active_buffers

    destroy_active_buffers()


@pytest.fixture(scope="session")
def dist_env():
    if "RANK" not in os.environ:
        pytest.skip("distributed kernel tests must be launched with torchrun")

    from kernel_test_utils import local_device_index

    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    device = local_device_index()
    torch.npu.set_device(device)

    yield rank, dist.get_world_size()

    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
