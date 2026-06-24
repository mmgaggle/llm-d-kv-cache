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

"""DOCA MEMOS storage backend (NVMe key-value store on BlueField via NIXL).

DOCA MEMOS is a NIXL backend (plugin name ``DOCA_MEMOS``) that stores KV-cache
blocks as values in an NVMe key-value namespace. Like the S3 ``OBJ`` backend it
stages GPU blocks through pinned CPU buffers and transfers ``DRAM -> OBJ``; the
differences are the plugin name, its init parameters, and a hard limit on the
object-key size (16 bytes). We therefore subclass :class:`ObjBackend` and only
override the parts that differ.
"""

import hashlib

import torch
from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger

from llmd_nixl.obj_backend import ObjBackend
from llmd_nixl.staged_backend import _StagedBackend

logger = get_logger(__name__)

# NIXL plugin name registered by the DOCA MEMOS backend (ai-dynamo/nixl #1717).
MEMOS_PLUGIN = "DOCA_MEMOS"

# Backend parameter, advertised by the backend via get_backend_params(), that
# reports the NVMe controller's maximum value size (bytes) per KV operation.
MAX_VALUE_SIZE_PARAM = "max_value_size"

# DOCA MEMOS keys are at most 16 bytes. NIXL decodes a hex metaInfo string of
# up to 32 chars into the binary key, and md5 produces exactly 16 bytes / 32
# hex chars, so we use the hex md5 of the file name as the object key.
DEFAULT_NUM_TASKS = 8192


def file_name_to_memos_key(name: str) -> str:
    """Map an arbitrary file name to a DOCA MEMOS object key.

    Returns a 32-character hex string (md5 digest); NIXL decodes it into the
    16-byte binary key DOCA expects.
    """
    return hashlib.md5(name.encode()).hexdigest()


def file_name_to_dev_id(name: str) -> int:
    """Stable per-object device id derived from the file name.

    DOCA MEMOS keys off the descriptor metaInfo (the hex key above); the dev_id
    is secondary but we keep it stable and non-zero per object.
    """
    return int(file_name_to_memos_key(name), 16) % (2**31)


def memos_backend_params(extra_config: dict | None) -> dict:
    """Build the ``create_backend("DOCA_MEMOS", ...)`` parameter dict.

    Shared by the transfer engine, the existence-lookup agent, and the
    max-value-size capability query so they all configure the device the same
    way. ``device_name`` is required.
    """
    cfg = extra_config or {}
    device_name = cfg.get("device_name")
    if not device_name:
        raise ValueError("DOCA MEMOS backend requires: device_name")

    params = {
        "device_name": device_name,
        "num_tasks": str(cfg.get("num_tasks", DEFAULT_NUM_TASKS)),
    }
    if cfg.get("nguid"):
        params["nguid"] = cfg["nguid"]
    if "ignore_read_not_found" in cfg:
        params["ignore_read_not_found"] = (
            "true" if _as_bool(cfg["ignore_read_not_found"]) else "false"
        )
    return params


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def query_max_value_size(extra_config: dict | None) -> int | None:
    """Return the device-advertised max value size (bytes), or ``None``.

    Spins up a throwaway NIXL agent, creates the DOCA MEMOS backend, and reads
    ``max_value_size`` from ``get_backend_params``. Returns ``None`` (and warns)
    if the backend does not advertise it or anything goes wrong, so callers can
    fall back to the configured block size.
    """
    try:
        params = memos_backend_params(extra_config)
        agent = nixl_agent("MemosCapabilityQuery", nixl_agent_config(backends=[]))
        agent.create_backend(MEMOS_PLUGIN, params)
        backend_params = agent.get_backend_params(MEMOS_PLUGIN) or {}
        raw = backend_params.get(MAX_VALUE_SIZE_PARAM)
        if not raw:
            logger.warning(
                "DOCA MEMOS backend did not advertise '%s'; "
                "block size will not be scaled to the device max value size",
                MAX_VALUE_SIZE_PARAM,
            )
            return None
        return int(raw)
    except Exception:
        logger.warning(
            "Failed to query DOCA MEMOS max value size; "
            "block size will not be scaled to the device max value size",
            exc_info=True,
        )
        return None


class MemosBackend(ObjBackend):
    """NIXL ``DOCA_MEMOS`` engine: stages GPU blocks and transfers DRAM -> OBJ.

    Reuses :class:`ObjBackend`'s pinned-buffer staging and descriptor plumbing;
    overrides construction (device params, not S3), key derivation, and the
    transfer entrypoints (to accept the handler's group/head-offset arguments).
    """

    nixl_source = "DRAM"
    nixl_dest = "OBJ"

    def __init__(
        self,
        io_threads: int,
        gpu_blocks_per_file: int,
        tensors: list[torch.Tensor],
        extra_config: dict | None = None,
    ):
        cfg = extra_config or {}
        self._memos_params = memos_backend_params(cfg)
        # head offsets (in GPU blocks) for the in-flight transfer, indexed by
        # file. Populated by async_store/load_gpu_blocks before _submit_transfer
        # and consumed by _build_nixl_file_entries. Safe because vLLM submits
        # from a single engine-core thread (see staged_backend notes).
        self._head_offsets: list[int] | None = None
        # Skip ObjBackend.__init__ (S3 validation / "OBJ" plugin); wire up the
        # staging machinery directly against the DOCA_MEMOS plugin instead.
        _StagedBackend.__init__(
            self, io_threads, gpu_blocks_per_file, tensors, MEMOS_PLUGIN
        )

    def _backend_params(self) -> dict:
        return dict(self._memos_params)

    # ------------------------------------------------------------------ #
    # 5-arg handler interface                                              #
    # ------------------------------------------------------------------ #
    # The storage handlers (llmd_fs_backend.worker) call transfer methods with
    # (job_id, group_indices, files, block_ids, head_offsets). The base
    # StorageOffloadEngine uses a 3-arg shape; override here so MEMOS matches
    # the live interface. group_indices is unused: like ObjBackend, MEMOS packs
    # all tensors (single KV-cache group) into one object per file.

    def async_store_gpu_blocks(  # type: ignore[override]
        self,
        job_id: int,
        group_indices: list,
        files: list[str],
        block_ids: list,
        head_offsets: list,
    ) -> bool:
        self.logger.debug("async_store_gpu_blocks in_flight=%d", len(self._transfers))
        self._head_offsets = head_offsets
        tensors, stagings = self._store_gpu_blocks_to_staging(block_ids)
        return self._submit_transfer(
            job_id, tensors, stagings, files, block_ids, "WRITE"
        )

    def async_load_gpu_blocks(  # type: ignore[override]
        self,
        job_id: int,
        group_indices: list,
        files: list[str],
        block_ids: list,
        head_offsets: list,
    ) -> bool:
        self.logger.debug("async_load_gpu_blocks in_flight=%d", len(self._transfers))
        self._head_offsets = head_offsets
        tensors, stagings = self._reserve_staging_for_load(block_ids)
        return self._submit_transfer(
            job_id, tensors, stagings, files, block_ids, "READ"
        )

    def _build_nixl_file_entries(self, fd_list, file_idx, block_list) -> list[tuple]:
        name = fd_list[file_idx]
        file_bytes = len(block_list) * len(self.tensors) * self._block_size
        # head_offsets[file_idx] is the offset (in GPU blocks) of the first
        # block within the object; non-zero only for a head-partial first file.
        head_blocks = self._head_offsets[file_idx] if self._head_offsets else 0
        file_offset = head_blocks * len(self.tensors) * self._block_size
        return [
            (
                file_offset,
                file_bytes,
                file_name_to_dev_id(name),
                file_name_to_memos_key(name),
            )
        ]
