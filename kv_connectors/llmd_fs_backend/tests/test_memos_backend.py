# Copyright 2025 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the DOCA MEMOS backend.

The full transfer engine needs CUDA and a NIXL build with the DOCA_MEMOS plugin
(plus BlueField hardware), so these tests cover the pure logic with NIXL mocked.
Tests that import the NIXL-dependent modules ``pytest.importorskip("nixl")`` so
they skip cleanly where NIXL is absent; the block-scaling tests inject a fake
``llmd_nixl.memos_backend`` module and need only vLLM + torch.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from llmd_fs_backend.spec import SharedStorageOffloadingSpec
from tests.test_spec import (
    GPU_BLOCK_SIZE,
    make_hybrid_kv_cache_config,
    make_vllm_config,
)

pytestmark = pytest.mark.no_cuda_required


# ---------------------------------------------------------------------------
# Pure helpers (need NIXL only because they live in the engine module)
# ---------------------------------------------------------------------------


def test_file_name_to_memos_key_is_16_byte_hex():
    pytest.importorskip("nixl")
    from llmd_nixl import memos_backend as mb

    key = mb.file_name_to_memos_key("base_r0/abc/de_g0/abcdef.bin")
    assert len(key) == 32  # 32 hex chars
    assert len(bytes.fromhex(key)) == 16  # decodes to a 16-byte DOCA key
    # deterministic and distinct per name
    assert key == mb.file_name_to_memos_key("base_r0/abc/de_g0/abcdef.bin")
    assert key != mb.file_name_to_memos_key("base_r0/abc/de_g0/ffffff.bin")
    assert 0 <= mb.file_name_to_dev_id("base_r0/abc/de_g0/abcdef.bin") < 2**31


def test_memos_backend_params_validation_and_defaults():
    pytest.importorskip("nixl")
    from llmd_nixl import memos_backend as mb

    with pytest.raises(ValueError, match="device_name"):
        mb.memos_backend_params({})

    params = mb.memos_backend_params({"device_name": "/dev/nvme0n1"})
    assert params == {"device_name": "/dev/nvme0n1", "num_tasks": "8192"}

    params = mb.memos_backend_params(
        {
            "device_name": "/dev/nvme0n1",
            "num_tasks": 256,
            "nguid": "0" * 32,
            "ignore_read_not_found": True,
        }
    )
    assert params["num_tasks"] == "256"  # coerced to str
    assert params["nguid"] == "0" * 32
    assert params["ignore_read_not_found"] == "true"


# ---------------------------------------------------------------------------
# Capability query (max value size)
# ---------------------------------------------------------------------------


def _patch_agent(monkeypatch, module, agent_cls):
    monkeypatch.setattr(module, "nixl_agent", agent_cls)
    monkeypatch.setattr(module, "nixl_agent_config", lambda **kw: None)


def test_query_max_value_size_reads_backend_params(monkeypatch):
    pytest.importorskip("nixl")
    from llmd_nixl import memos_backend as mb

    class FakeAgent:
        def __init__(self, *a, **k):
            pass

        def create_backend(self, name, params):
            assert name == "DOCA_MEMOS"

        def get_backend_params(self, name):
            return {"max_value_size": "1048576"}

    _patch_agent(monkeypatch, mb, FakeAgent)
    assert mb.query_max_value_size({"device_name": "/dev/nvme0n1"}) == 1048576


def test_query_max_value_size_none_when_absent(monkeypatch):
    pytest.importorskip("nixl")
    from llmd_nixl import memos_backend as mb

    class FakeAgent:
        def __init__(self, *a, **k):
            pass

        def create_backend(self, name, params):
            pass

        def get_backend_params(self, name):
            return {}  # backend did not advertise max_value_size

    _patch_agent(monkeypatch, mb, FakeAgent)
    assert mb.query_max_value_size({"device_name": "/dev/nvme0n1"}) is None


def test_query_max_value_size_none_on_error(monkeypatch):
    pytest.importorskip("nixl")
    from llmd_nixl import memos_backend as mb

    class FakeAgent:
        def __init__(self, *a, **k):
            raise RuntimeError("no device")

    _patch_agent(monkeypatch, mb, FakeAgent)
    assert mb.query_max_value_size({"device_name": "/dev/nvme0n1"}) is None


# ---------------------------------------------------------------------------
# Existence lookup
# ---------------------------------------------------------------------------


def test_memos_lookup_uses_actual_query_mode(monkeypatch):
    pytest.importorskip("nixl")
    from llmd_nixl import nixl_lookup

    seen = {}

    class FakeAgent:
        def __init__(self, *a, **k):
            pass

        def create_backend(self, name, params):
            seen["backend"] = name
            seen["params"] = params

        def query_memory(self, descs, backend, mem_type):
            seen["query"] = (descs, backend, mem_type)
            return [object()]  # not None -> exists

    _patch_agent(monkeypatch, nixl_lookup, FakeAgent)

    lookup = nixl_lookup.MemosLookup({"device_name": "/dev/nvme0n1"})
    assert seen["backend"] == "DOCA_MEMOS"
    assert seen["params"]["query_mem_mode"] == "actual"

    assert lookup.exists("base_r0/abc/de_g0/abcdef.bin") is True
    _descs, backend, mem_type = seen["query"]
    assert backend == "DOCA_MEMOS"
    assert mem_type == "OBJ"


def test_memos_lookup_missing_key_returns_false(monkeypatch):
    pytest.importorskip("nixl")
    from llmd_nixl import nixl_lookup

    class FakeAgent:
        def __init__(self, *a, **k):
            pass

        def create_backend(self, name, params):
            pass

        def query_memory(self, descs, backend, mem_type):
            return [None]  # not found

    _patch_agent(monkeypatch, nixl_lookup, FakeAgent)
    assert nixl_lookup.MemosLookup({"device_name": "/dev/nvme0n1"}).exists("k") is False


# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend, expected",
    [("MEMOS", "MemosBackend"), ("OBJ", "ObjBackend"), (None, "ObjBackend")],
)
def test_create_engine_selects_backend(monkeypatch, backend, expected):
    pytest.importorskip("nixl")
    from llmd_nixl import worker as nixl_worker

    made = {}

    def make_stub(name):
        def stub(**kwargs):
            made["cls"] = name
            made["kwargs"] = kwargs
            return SimpleNamespace(**kwargs)

        return stub

    monkeypatch.setattr(nixl_worker, "MemosBackend", make_stub("MemosBackend"))
    monkeypatch.setattr(nixl_worker, "ObjBackend", make_stub("ObjBackend"))

    handlers = object.__new__(nixl_worker.NixlStorageOffloadingHandlers)
    kv_caches = SimpleNamespace(tensors=[SimpleNamespace(tensor="t0")])
    extra_config = {} if backend is None else {"backend": backend}

    handlers._create_engine(
        io_threads=4,
        gpu_blocks_per_file=2,
        kv_caches=kv_caches,
        read_preferring_workers=1,
        max_write_queued_seconds=30.0,
        extra_config=extra_config,
        gds_mode="disabled",
    )
    assert made["cls"] == expected
    assert made["kwargs"]["tensors"] == ["t0"]


# ---------------------------------------------------------------------------
# Block-size scaling to the device max value size
# ---------------------------------------------------------------------------


def _expected_per_block_bytes(kv_cache_config) -> int:
    """Mirror SharedStorageOffloadingSpec._memos_object_bytes_per_block."""
    return sum(
        g.kv_cache_spec.page_size_bytes
        * GPU_BLOCK_SIZE
        // g.kv_cache_spec.block_size
        * len(g.layer_names)
        for g in kv_cache_config.kv_cache_groups
    )


def _make_memos_spec(tmp_path, monkeypatch, max_value_size, block_size=256):
    kv_cache_config = make_hybrid_kv_cache_config()

    fake = ModuleType("llmd_nixl.memos_backend")
    fake.query_max_value_size = lambda cfg: max_value_size
    monkeypatch.setitem(sys.modules, "llmd_nixl.memos_backend", fake)

    extra_config = {
        "shared_storage_path": str(tmp_path),
        "backend": "MEMOS",
        "device_name": "/dev/nvme0n1",
        "block_size": block_size,
    }
    spec = SharedStorageOffloadingSpec(make_vllm_config(extra_config), kv_cache_config)
    return spec, kv_cache_config


def test_block_size_scaled_down_to_fit_max_value(tmp_path, monkeypatch):
    # Pick a cap that fits exactly 5 GPU blocks; request 256/16 = 16.
    per_block = _expected_per_block_bytes(make_hybrid_kv_cache_config())
    spec, _ = _make_memos_spec(tmp_path, monkeypatch, max_value_size=per_block * 5)
    assert spec.gpu_blocks_per_file == 5
    assert spec.offloaded_block_size == 5 * GPU_BLOCK_SIZE
    assert spec.block_size_factor == 5


def test_block_size_unchanged_when_object_fits(tmp_path, monkeypatch):
    per_block = _expected_per_block_bytes(make_hybrid_kv_cache_config())
    spec, _ = _make_memos_spec(tmp_path, monkeypatch, max_value_size=per_block * 1000)
    assert spec.gpu_blocks_per_file == 256 // GPU_BLOCK_SIZE
    assert spec.block_size_factor == spec.gpu_blocks_per_file


def test_block_size_unchanged_when_cap_unavailable(tmp_path, monkeypatch):
    spec, _ = _make_memos_spec(tmp_path, monkeypatch, max_value_size=None)
    assert spec.gpu_blocks_per_file == 256 // GPU_BLOCK_SIZE
    assert spec.block_size_factor == spec.gpu_blocks_per_file
