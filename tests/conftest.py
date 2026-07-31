import os

import pytest
import torch
import torch.distributed as dist


@pytest.fixture(autouse=True)
def cleanup_moonep_buffers():
    yield
    from tests.kernel_test_utils import destroy_active_buffers

    destroy_active_buffers()


@pytest.fixture(scope="session")
def dist_env():
    if "RANK" not in os.environ:
        pytest.skip("distributed kernel tests must be launched with torchrun")

    from tests.kernel_test_utils import local_device_index

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = local_device_index()
    torch.cuda.set_device(device)

    yield rank, dist.get_world_size()

    dist.barrier(device_ids=[device])
    if dist.is_initialized():
        dist.destroy_process_group()
