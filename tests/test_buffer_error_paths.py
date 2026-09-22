"""Unit tests for the resource-cleanup fix in moonep/buffer.py (issue #17).

Unlike the rest of the test suite, these do NOT require GPU/NVLink or
torchrun: they stub out `moonep._C` with plain Python mocks so the pure
control-flow (which fds/handles get closed/released when
`_exchange_ipc_fds` raises) can be exercised on any machine.

Run with:
    python -m pytest tests/test_buffer_error_paths.py -v
"""

import importlib.util
import os
import sys
import types
from unittest import mock

import pytest

_BUFFER_PY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "moonep",
    "buffer.py",
)


@pytest.fixture
def buffer_module():
    """Load moonep/buffer.py directly, with a stubbed-out moonep._C.

    Importing the `moonep` package itself pulls in moonep.api ->
    moonep.planning, which needs the CUDA toolkit's cuda-python bindings
    and is irrelevant to what's under test here, so buffer.py is loaded
    as a standalone module (bypassing moonep/__init__.py) instead of via
    `import moonep.buffer`.
    """
    fake_c = types.SimpleNamespace(
        FABRIC_HANDLE_BYTES=64,
        nvl_dist_alloc=mock.Mock(),
        nvl_dist_map=mock.Mock(),
        nvl_fabric_supported=mock.Mock(return_value=False),
        nvl_release_mem_handle=mock.Mock(),
        get_vmm_granularity=mock.Mock(return_value=1),
        get_multicast_granularity=mock.Mock(return_value=1),
        nvl_multicast_supported=mock.Mock(return_value=True),
        nvl_multicast_create=mock.Mock(),
        nvl_multicast_import=mock.Mock(),
        nvl_multicast_add_device=mock.Mock(),
        nvl_multicast_bind_map=mock.Mock(),
    )
    with mock.patch.dict(sys.modules, {"moonep._C": fake_c}):
        spec = importlib.util.spec_from_file_location(
            "moonep_buffer_under_test", _BUFFER_PY
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod


class _FakeTensor:
    """Stand-in for the 0-d int64 tensors buffer.py calls .item() on."""

    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


def test_map_nvl_dist_tensor_closes_local_fd_when_exchange_raises(buffer_module):
    """_map_nvl_dist_tensor (fd path): local_fd must be closed even if
    _exchange_ipc_fds raises, not just on the success path."""
    with mock.patch.object(
        buffer_module, "_exchange_ipc_fds", side_effect=RuntimeError("boom")
    ), mock.patch.object(buffer_module.os, "close") as mock_close:
        with pytest.raises(RuntimeError, match="boom"):
            buffer_module._map_nvl_dist_tensor(
                chunk_shape=[1],
                dtype=None,
                shareable=_FakeTensor(42),
                keepalive=None,
                local_rank=0,
                world_size=2,
                group=None,
                use_fabric=False,
            )
        mock_close.assert_called_once_with(42)


def test_multicast_view_releases_root_handle_when_exchange_raises(buffer_module):
    """_create_nvl_multicast_view (fd path, root): if _exchange_ipc_fds
    raises after nvl_multicast_create already handed the root a live
    handle, that handle must be released, not leaked."""
    buffer_module.nvl_multicast_create.return_value = (777, _FakeTensor(11))

    with mock.patch.object(
        buffer_module, "_exchange_ipc_fds", side_effect=RuntimeError("boom")
    ), mock.patch.object(buffer_module.os, "close") as mock_close:
        with pytest.raises(RuntimeError, match="boom"):
            buffer_module._create_nvl_multicast_view(
                meta_buf=mock.Mock(numel=lambda: 8, element_size=lambda: 2),
                owned_handle=999,
                local_rank=0,  # root
                world_size=2,
                group=None,
                use_fabric=False,
            )
        mock_close.assert_called_once_with(11)
        buffer_module.nvl_release_mem_handle.assert_called_once_with(777)


def test_multicast_view_closes_root_fd_when_import_raises(buffer_module):
    """_create_nvl_multicast_view (fd path, non-root): the imported root_fd
    must still be closed if nvl_multicast_import raises."""
    with mock.patch.object(
        buffer_module, "_exchange_ipc_fds", return_value={0: 55}
    ), mock.patch.object(
        buffer_module, "nvl_multicast_import", side_effect=RuntimeError("boom")
    ), mock.patch.object(buffer_module.os, "close") as mock_close:
        with pytest.raises(RuntimeError, match="boom"):
            buffer_module._create_nvl_multicast_view(
                meta_buf=mock.Mock(numel=lambda: 8, element_size=lambda: 2),
                owned_handle=999,
                local_rank=1,  # not root
                world_size=2,
                group=None,
                use_fabric=False,
            )
        mock_close.assert_called_once_with(55)
        # Non-root never created a handle of its own, so nothing to release.
        buffer_module.nvl_release_mem_handle.assert_not_called()


def test_single_owner_tensor_closes_local_fd_when_exchange_raises(buffer_module):
    """create_nvl_single_owner_tensor (fd path, owner): local_fd must be
    closed even if _exchange_ipc_fds raises."""
    buffer_module.nvl_dist_alloc.return_value = (
        mock.Mock(),
        _FakeTensor(33),
        888,
    )

    with mock.patch.object(
        buffer_module, "_exchange_ipc_fds", side_effect=RuntimeError("boom")
    ), mock.patch.object(buffer_module.os, "close") as mock_close, mock.patch.object(
        buffer_module.dist, "get_world_size", return_value=2
    ):
        with pytest.raises(RuntimeError, match="boom"):
            buffer_module.create_nvl_single_owner_tensor(
                shape=[1],
                dtype=None,
                owner_rank=0,
                local_rank=0,  # owner
                group=None,
            )
        mock_close.assert_called_once_with(33)
