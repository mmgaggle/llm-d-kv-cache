# DOCA MEMOS Guide

The DOCA MEMOS backend offloads KV-cache blocks to an NVMe key-value (KV) namespace
accelerated by NVIDIA BlueField DPUs. Each offloaded block is stored as a single KV
*value* keyed by the block hash.

This backend is built on top of NIXL, using the `DOCA_MEMOS` plugin
(NVIDIA's DOCA KV library). It transfers `DRAM -> OBJ` exactly like the
[object store](./object_store.md) backend, staging GPU blocks through pinned CPU buffers.

## Requirements

- NIXL built with the `DOCA_MEMOS` plugin (installed by the Python wheel during the build)
- DOCA SDK 4.4 or later (`libdoca_kv`, `libdoca_common`, `libdoca_nvme_kernel_kvdev`)
- An NVIDIA BlueField-3 DPU (or newer) exposing an NVMe KV device (e.g. `/dev/nvme0n1`)

## Build

Follow the standard [build instructions](../README.md#installation).

## Configuration

Set `backend: "MEMOS"` and the device parameters in `kv_connector_extra_config` in your
vLLM config:

```yaml
--kv-transfer-config '{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "SharedStorageOffloadingSpec",
    "spec_module_path": "llmd_fs_backend.spec",
    "shared_storage_path": "/mnt/nvme/kv-cache/",
    "block_size": 256,
    "threads_per_gpu": "64",
    "backend": "MEMOS",
    "device_name": "/dev/nvme0n1",
    "num_tasks": "8192",
    "nguid": "00000000000000000000000000000000"
  }
}'
--distributed_executor_backend "mp"
```

`shared_storage_path` is still used to derive the per-block key names (the path is hashed
into a 16-byte object key); no files are written to it.

### Backend parameters

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `device_name` | yes | Path to the NVMe KV device | — |
| `num_tasks` | no | DOCA task-pool size (clamped to the device max) | `8192` |
| `nguid` | no | 32-char hex NVMe namespace NGUID | all zeros |
| `ignore_read_not_found` | no | If `true`, reads of missing keys succeed (buffer undefined) instead of failing | `false` |

Existence lookups (the indexer's read path) issue real DOCA `EXIST` operations: the
backend's lookup agent is created with `query_mem_mode=actual`, overriding the backend's
default `assume_success` (which would report every key as present).

## Block sizing and the device max value size

An NVMe KV controller advertises a maximum value size per KV operation. Because each
offloaded object packs `block_size / hash_block_size` GPU blocks (across all layers) into
one value, the connector reads the advertised limit via NIXL
(`get_backend_params("DOCA_MEMOS")["max_value_size"]`) and automatically lowers the number
of GPU blocks per object so the packed value fits. This runs on every rank during startup,
so the scheduler and workers agree on the resulting layout.

If the NIXL build does not advertise a max value size, the connector logs a warning and
uses the configured `block_size` unchanged — make sure your `block_size` is small enough
for the device in that case.

## Manual smoke test

This backend requires BlueField hardware and a NIXL build with the `DOCA_MEMOS` plugin, so
it cannot be exercised by the CPU unit tests. To validate end to end on a suitable host,
run vLLM with the config above and confirm a store → lookup → load round-trip:

1. Send a request long enough to fill at least one offloaded block; confirm
   `PUT`/store transfers complete in the worker logs.
2. Re-send the same prefix and confirm the indexer reports a cache hit (lookup `EXIST`
   succeeds) and `GET`/load transfers complete.
3. Watch startup logs for the `DOCA MEMOS: scaling offloaded block size ...` /
   `... fits device max value size` message to confirm block sizing against the device.
