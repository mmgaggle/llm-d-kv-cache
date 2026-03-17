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

"""Tests for GPU↔S3 worker transfer handlers."""

import io
import time
import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock, patch, PropertyMock



def make_gpu_tensor(num_blocks=100, num_heads=8, block_size=16, head_dim=256):
    """Create a CPU tensor shaped like a KV cache (simulating GPU)."""
    return torch.randn(2, num_blocks, num_heads, block_size, head_dim, dtype=torch.float16)


def make_kv_caches(num_layers=2, **kwargs):
    """Create dict of mock KV cache tensors."""
    return {f"layer_{i}": make_gpu_tensor(**kwargs) for i in range(num_layers)}


def serialize_blocks(blocks_data):
    """Serialize block data the same way the PUT handler does."""
    buf = io.BytesIO()
    np.savez_compressed(buf, *[b.numpy() for b in blocks_data])
    return buf.getvalue()


def _make_concrete_put_handler_cls():
    """
    Create a concrete subclass of GPUS3OffloadingHandler.

    GPUS3OffloadingHandler inherits `wait` as abstract from OffloadingHandler
    but doesn't implement it (only S3GPUOffloadingHandler does).
    For unit tests we supply a trivial stub.
    """
    from llmd_s3_backend.worker import GPUS3OffloadingHandler

    class ConcretePutHandler(GPUS3OffloadingHandler):
        def wait(self, job_id: int) -> bool:
            with self.lock:
                future = self.pending_futures.get(job_id)
            if future is None:
                return True
            try:
                future.result(timeout=30.0)
                return True
            except Exception:
                return False

    return ConcretePutHandler


# ---------------------------------------------------------------------------
# S3OffloadingHandler (base class, tested via GPUS3OffloadingHandler)
# ---------------------------------------------------------------------------
class TestS3OffloadingHandlerInit:
    """Test base handler initialization via concrete PUT handler subclass."""

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_basic_attributes(self, mock_client_cls, mock_stream, mock_attn_backends):
        HandlerCls = _make_concrete_put_handler_cls()
        kv_caches = make_kv_caches(num_layers=2)
        handler = HandlerCls(
            model_name="llama-7b",
            tp_size=4,
            tp_rank=1,
            kv_caches=kv_caches,
            gpu_blocks_per_file=256,
            attn_backends=mock_attn_backends,
            dtype=torch.float16,
            bucket="my-bucket",
            prefix="kv-cache",
            region="us-west-2",
            threads_per_gpu=32,
        )
        assert handler.model_name == "llama-7b"
        assert handler.tp_size == 4
        assert handler.tp_rank == 1
        assert handler.dtype == torch.float16
        assert handler.gpu_blocks_per_file == 256
        assert handler.threads_per_gpu == 32

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_threads_capped_at_max(self, mock_client_cls, mock_stream, mock_attn_backends):
        from llmd_s3_backend.worker import DEFAULT_MAX_THREADS_PER_GPU
        HandlerCls = _make_concrete_put_handler_cls()

        kv_caches = make_kv_caches(num_layers=1)
        handler = HandlerCls(
            model_name="m",
            tp_size=1,
            tp_rank=0,
            kv_caches=kv_caches,
            gpu_blocks_per_file=1,
            attn_backends=mock_attn_backends,
            dtype=torch.float16,
            bucket="b",
            threads_per_gpu=999,
        )
        assert handler.threads_per_gpu == DEFAULT_MAX_THREADS_PER_GPU

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_base_key_format(self, mock_client_cls, mock_stream, mock_attn_backends):
        HandlerCls = _make_concrete_put_handler_cls()
        kv_caches = make_kv_caches(num_layers=1)
        handler = HandlerCls(
            model_name="gpt2",
            tp_size=2,
            tp_rank=0,
            kv_caches=kv_caches,
            gpu_blocks_per_file=1,
            attn_backends=mock_attn_backends,
            dtype=torch.bfloat16,
            bucket="b",
            prefix="my-prefix",
        )
        assert handler.base_key == "my-prefix/gpt2/tp_2/rank_0/bfloat16"


class TestGetS3Key:
    """Test S3 key generation from block hashes."""

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_integer_block_hash(self, mock_client_cls, mock_stream, mock_attn_backends):
        HandlerCls = _make_concrete_put_handler_cls()
        kv_caches = make_kv_caches(num_layers=1)
        handler = HandlerCls(
            model_name="m", tp_size=1, tp_rank=0,
            kv_caches=kv_caches, gpu_blocks_per_file=1,
            attn_backends=mock_attn_backends, dtype=torch.float16,
            bucket="b", prefix="p",
        )
        key = handler._get_s3_key(0xABCDE12345678900)
        assert key == "p/m/tp_1/rank_0/float16/abc/de/abcde12345678900.bin"

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_bytes_block_hash(self, mock_client_cls, mock_stream, mock_attn_backends):
        HandlerCls = _make_concrete_put_handler_cls()
        kv_caches = make_kv_caches(num_layers=1)
        handler = HandlerCls(
            model_name="m", tp_size=1, tp_rank=0,
            kv_caches=kv_caches, gpu_blocks_per_file=1,
            attn_backends=mock_attn_backends, dtype=torch.float16,
            bucket="b", prefix="p",
        )
        block_hash_bytes = (0x00FFEEDDCCBBAA99).to_bytes(8, "little")
        key = handler._get_s3_key(block_hash_bytes)
        assert key.endswith(".bin")
        assert key.startswith("p/m/tp_1/rank_0/float16/")

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_key_has_prefix_structure(self, mock_client_cls, mock_stream, mock_attn_backends):
        HandlerCls = _make_concrete_put_handler_cls()
        kv_caches = make_kv_caches(num_layers=1)
        handler = HandlerCls(
            model_name="m", tp_size=1, tp_rank=0,
            kv_caches=kv_caches, gpu_blocks_per_file=1,
            attn_backends=mock_attn_backends, dtype=torch.float16,
            bucket="b", prefix="p",
        )
        key = handler._get_s3_key(42)
        parts = key.split("/")
        # prefix/model/tp_N/rank_N/dtype/subfolder1/subfolder2/hash.bin
        assert len(parts) == 8


class TestGetFinished:
    """Test polling completed transfers."""

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_empty_when_nothing_completed(self, mock_client_cls, mock_stream, mock_attn_backends):
        HandlerCls = _make_concrete_put_handler_cls()
        kv_caches = make_kv_caches(num_layers=1)
        handler = HandlerCls(
            model_name="m", tp_size=1, tp_rank=0,
            kv_caches=kv_caches, gpu_blocks_per_file=1,
            attn_backends=mock_attn_backends, dtype=torch.float16, bucket="b",
        )
        assert handler.get_finished() == []

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_returns_and_clears_completed(self, mock_client_cls, mock_stream, mock_attn_backends):
        HandlerCls = _make_concrete_put_handler_cls()
        kv_caches = make_kv_caches(num_layers=1)
        handler = HandlerCls(
            model_name="m", tp_size=1, tp_rank=0,
            kv_caches=kv_caches, gpu_blocks_per_file=1,
            attn_backends=mock_attn_backends, dtype=torch.float16, bucket="b",
        )
        handler.completed_jobs = [(1, True), (2, False)]

        results = handler.get_finished()
        assert len(results) == 2
        assert (1, True) in results
        assert (2, False) in results

        # Second call should be empty
        assert handler.get_finished() == []


class TestGetKvCacheParameters:
    """Test KV cache layout detection."""

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_kv_before_num_blocks(self, mock_client_cls, mock_stream):
        HandlerCls = _make_concrete_put_handler_cls()

        # Backend returns (2, num_blocks, heads, block_size, head_dim)
        backend = Mock()
        backend.get_kv_cache_shape = Mock(return_value=(2, 1234, 8, 16, 256))

        # GPU tensor has matching shape
        gpu_tensor = Mock()
        gpu_tensor.shape = (2, 500, 8, 16, 256)

        kv_caches = make_kv_caches(num_layers=1)
        handler = HandlerCls(
            model_name="m", tp_size=1, tp_rank=0,
            kv_caches=kv_caches, gpu_blocks_per_file=1,
            attn_backends={"layer_0": backend}, dtype=torch.float16, bucket="b",
        )

        idx, kv_before, layers_before = handler.get_kv_cache_parameters(
            {"layer_0": gpu_tensor}
        )
        assert idx == [1]
        assert kv_before == [True]
        assert layers_before == [True]

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_num_blocks_first(self, mock_client_cls, mock_stream):
        HandlerCls = _make_concrete_put_handler_cls()

        # Backend returns (num_blocks, ...)
        backend = Mock()
        backend.get_kv_cache_shape = Mock(return_value=(1234, 8, 16, 256))

        gpu_tensor = Mock()
        gpu_tensor.shape = (500, 8, 16, 256)

        kv_caches = make_kv_caches(num_layers=1)
        handler = HandlerCls(
            model_name="m", tp_size=1, tp_rank=0,
            kv_caches=kv_caches, gpu_blocks_per_file=1,
            attn_backends={"layer_0": backend}, dtype=torch.float16, bucket="b",
        )

        idx, kv_before, layers_before = handler.get_kv_cache_parameters(
            {"layer_0": gpu_tensor}
        )
        assert idx == [0]
        assert kv_before == [False]
        assert layers_before == [True]


# ---------------------------------------------------------------------------
# GPUS3OffloadingHandler (PUT path)
# ---------------------------------------------------------------------------
class TestPutBlocksToS3:
    """Test _put_blocks_to_s3 serialization and upload."""

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_successful_upload(self, mock_client_cls, mock_stream, mock_attn_backends):
        HandlerCls = _make_concrete_put_handler_cls()
        kv_caches = make_kv_caches(num_layers=2, num_blocks=50)
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        handler = HandlerCls(
            model_name="m", tp_size=1, tp_rank=0,
            kv_caches=kv_caches, gpu_blocks_per_file=4,
            attn_backends=mock_attn_backends, dtype=torch.float16,
            bucket="b",
        )

        job_id, success = handler._put_blocks_to_s3(
            job_id=1,
            s3_key="test/key.bin",
            tensors=handler.src_tensors,
            block_ids=[0, 1, 2, 3],
        )
        assert job_id == 1
        assert success is True
        mock_client.put_object.assert_called_once()

        # Verify uploaded data is valid numpy archive
        uploaded_data = mock_client.put_object.call_args[0][1]
        loaded = np.load(io.BytesIO(uploaded_data))
        assert len(loaded.files) == 2  # 2 layers

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_upload_failure_returns_false(self, mock_client_cls, mock_stream, mock_attn_backends):
        HandlerCls = _make_concrete_put_handler_cls()
        kv_caches = make_kv_caches(num_layers=1, num_blocks=10)
        mock_client = MagicMock()
        mock_client.put_object.side_effect = RuntimeError("network error")
        mock_client_cls.return_value = mock_client

        handler = HandlerCls(
            model_name="m", tp_size=1, tp_rank=0,
            kv_caches=kv_caches, gpu_blocks_per_file=4,
            attn_backends=mock_attn_backends, dtype=torch.float16,
            bucket="b",
        )

        job_id, success = handler._put_blocks_to_s3(
            job_id=7,
            s3_key="test/key.bin",
            tensors=handler.src_tensors,
            block_ids=[0, 1],
        )
        assert job_id == 7
        assert success is False


class TestGPUS3TransferAsync:
    """Test async PUT transfer submission."""

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_empty_spec_returns_true(self, mock_client_cls, mock_stream, mock_attn_backends):
        HandlerCls = _make_concrete_put_handler_cls()
        kv_caches = make_kv_caches(num_layers=1, num_blocks=10)
        handler = HandlerCls(
            model_name="m", tp_size=1, tp_rank=0,
            kv_caches=kv_caches, gpu_blocks_per_file=4,
            attn_backends=mock_attn_backends, dtype=torch.float16,
            bucket="b",
        )

        # dst_spec with no block_hashes
        src_spec = Mock()
        src_spec.block_ids = [0, 1, 2]
        dst_spec = Mock()
        dst_spec.block_hashes = []

        result = handler.transfer_async(1, (src_spec, dst_spec))
        assert result is True

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_none_dst_spec_returns_true(self, mock_client_cls, mock_stream, mock_attn_backends):
        HandlerCls = _make_concrete_put_handler_cls()
        kv_caches = make_kv_caches(num_layers=1, num_blocks=10)
        handler = HandlerCls(
            model_name="m", tp_size=1, tp_rank=0,
            kv_caches=kv_caches, gpu_blocks_per_file=4,
            attn_backends=mock_attn_backends, dtype=torch.float16,
            bucket="b",
        )

        src_spec = Mock()
        result = handler.transfer_async(1, (src_spec, None))
        assert result is True

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_submits_futures(self, mock_client_cls, mock_stream, mock_attn_backends):
        HandlerCls = _make_concrete_put_handler_cls()
        kv_caches = make_kv_caches(num_layers=1, num_blocks=20)
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        handler = HandlerCls(
            model_name="m", tp_size=1, tp_rank=0,
            kv_caches=kv_caches, gpu_blocks_per_file=4,
            attn_backends=mock_attn_backends, dtype=torch.float16,
            bucket="b",
        )

        src_spec = Mock()
        src_spec.block_ids = list(range(8))
        dst_spec = Mock()
        dst_spec.block_hashes = [0xAABB, 0xCCDD]

        result = handler.transfer_async(42, (src_spec, dst_spec))
        assert result is True

        # Should have submitted jobs to executor; wait for completion
        handler.executor.shutdown(wait=True)

        # Should have pending futures
        assert 42 in handler.pending_futures


# ---------------------------------------------------------------------------
# S3GPUOffloadingHandler (GET path)
# ---------------------------------------------------------------------------
class TestGetBlocksFromS3:
    """Test _get_blocks_from_s3 download and deserialization."""

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_successful_download(self, mock_client_cls, mock_stream, mock_attn_backends):
        from llmd_s3_backend.worker import S3GPUOffloadingHandler

        kv_caches = make_kv_caches(num_layers=2, num_blocks=50)
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Prepare serialized data that matches what GET expects
        block_ids = [0, 1]
        blocks_data = [t[:, block_ids, :, :, :] for t in kv_caches.values()]
        serialized = serialize_blocks(blocks_data)
        mock_client.get_object.return_value = serialized

        handler = S3GPUOffloadingHandler(
            model_name="m", tp_size=1, tp_rank=0, dtype=torch.float16,
            gpu_blocks_per_file=4, kv_caches=kv_caches,
            attn_backends=mock_attn_backends, bucket="b",
        )

        job_id, success = handler._get_blocks_from_s3(
            job_id=1,
            s3_key="test/key.bin",
            tensors=handler.dst_tensors,
            block_ids=block_ids,
        )
        assert job_id == 1
        assert success is True
        mock_client.get_object.assert_called_once_with("test/key.bin")

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_download_failure_returns_false(self, mock_client_cls, mock_stream, mock_attn_backends):
        from llmd_s3_backend.worker import S3GPUOffloadingHandler

        kv_caches = make_kv_caches(num_layers=1, num_blocks=10)
        mock_client = MagicMock()
        mock_client.get_object.side_effect = RuntimeError("connection refused")
        mock_client_cls.return_value = mock_client

        handler = S3GPUOffloadingHandler(
            model_name="m", tp_size=1, tp_rank=0, dtype=torch.float16,
            gpu_blocks_per_file=4, kv_caches=kv_caches,
            attn_backends=mock_attn_backends, bucket="b",
        )

        job_id, success = handler._get_blocks_from_s3(
            job_id=5,
            s3_key="test/key.bin",
            tensors=handler.dst_tensors,
            block_ids=[0],
        )
        assert job_id == 5
        assert success is False

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_404_triggers_cache_invalidation(self, mock_client_cls, mock_stream, mock_attn_backends):
        from llmd_s3_backend.worker import S3GPUOffloadingHandler

        kv_caches = make_kv_caches(num_layers=1, num_blocks=10)
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Simulate boto3 ClientError with 404
        error = Exception("Not Found")
        error.response = {"Error": {"Code": "404"}}
        mock_client.get_object.side_effect = error

        handler = S3GPUOffloadingHandler(
            model_name="m", tp_size=1, tp_rank=0, dtype=torch.float16,
            gpu_blocks_per_file=4, kv_caches=kv_caches,
            attn_backends=mock_attn_backends, bucket="b",
        )

        mock_manager = Mock()
        handler.manager = mock_manager

        job_id, success = handler._get_blocks_from_s3(
            job_id=3,
            s3_key="missing/key.bin",
            tensors=handler.dst_tensors,
            block_ids=[0],
        )
        assert success is False
        mock_manager.invalidate_cache_entry.assert_called_once_with("missing/key.bin")

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_data_integrity(self, mock_client_cls, mock_stream, mock_attn_backends):
        """Verify that data downloaded from S3 is correctly written to tensors."""
        from llmd_s3_backend.worker import S3GPUOffloadingHandler

        kv_caches = make_kv_caches(num_layers=1, num_blocks=10)
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Create known data for blocks 0 and 1
        block_ids = [0, 1]
        original_tensors = list(kv_caches.values())
        source_data = [t[:, block_ids, :, :, :].clone() for t in original_tensors]
        serialized = serialize_blocks(source_data)
        mock_client.get_object.return_value = serialized

        # Zero out the target blocks before download
        for t in original_tensors:
            t[:, block_ids, :, :, :] = 0

        handler = S3GPUOffloadingHandler(
            model_name="m", tp_size=1, tp_rank=0, dtype=torch.float16,
            gpu_blocks_per_file=4, kv_caches=kv_caches,
            attn_backends=mock_attn_backends, bucket="b",
        )

        handler._get_blocks_from_s3(
            job_id=1,
            s3_key="key.bin",
            tensors=handler.dst_tensors,
            block_ids=block_ids,
        )

        # Blocks should now contain the source data
        for dst, src in zip(handler.dst_tensors, source_data):
            torch.testing.assert_close(
                dst[:, block_ids, :, :, :],
                src.to(dtype=dst.dtype),
            )


class TestS3GPUTransferAsync:
    """Test async GET transfer submission."""

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_empty_src_spec_returns_true(self, mock_client_cls, mock_stream, mock_attn_backends):
        from llmd_s3_backend.worker import S3GPUOffloadingHandler

        kv_caches = make_kv_caches(num_layers=1, num_blocks=10)
        handler = S3GPUOffloadingHandler(
            model_name="m", tp_size=1, tp_rank=0, dtype=torch.float16,
            gpu_blocks_per_file=4, kv_caches=kv_caches,
            attn_backends=mock_attn_backends, bucket="b",
        )

        src_spec = Mock()
        src_spec.block_hashes = []
        dst_spec = Mock()
        dst_spec.block_ids = [0, 1]

        result = handler.transfer_async(1, (src_spec, dst_spec))
        assert result is True

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_none_src_spec_returns_true(self, mock_client_cls, mock_stream, mock_attn_backends):
        from llmd_s3_backend.worker import S3GPUOffloadingHandler

        kv_caches = make_kv_caches(num_layers=1, num_blocks=10)
        handler = S3GPUOffloadingHandler(
            model_name="m", tp_size=1, tp_rank=0, dtype=torch.float16,
            gpu_blocks_per_file=4, kv_caches=kv_caches,
            attn_backends=mock_attn_backends, bucket="b",
        )

        dst_spec = Mock()
        result = handler.transfer_async(1, (None, dst_spec))
        assert result is True

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_submits_get_jobs(self, mock_client_cls, mock_stream, mock_attn_backends):
        from llmd_s3_backend.worker import S3GPUOffloadingHandler

        kv_caches = make_kv_caches(num_layers=1, num_blocks=20)
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Prepare valid download data
        block_ids = [0, 1, 2, 3]
        blocks_data = [t[:, block_ids, :, :, :] for t in kv_caches.values()]
        serialized = serialize_blocks(blocks_data)
        mock_client.get_object.return_value = serialized

        handler = S3GPUOffloadingHandler(
            model_name="m", tp_size=1, tp_rank=0, dtype=torch.float16,
            gpu_blocks_per_file=4, kv_caches=kv_caches,
            attn_backends=mock_attn_backends, bucket="b",
        )

        src_spec = Mock()
        src_spec.block_hashes = [0xAABB]
        dst_spec = Mock()
        dst_spec.block_ids = [0, 1, 2, 3]

        result = handler.transfer_async(10, (src_spec, dst_spec))
        assert result is True

        handler.executor.shutdown(wait=True)
        assert 10 in handler.pending_futures


class TestS3GPUWait:
    """Test blocking wait for job completion."""

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_wait_nonexistent_job_returns_true(self, mock_client_cls, mock_stream, mock_attn_backends):
        from llmd_s3_backend.worker import S3GPUOffloadingHandler

        kv_caches = make_kv_caches(num_layers=1, num_blocks=10)
        handler = S3GPUOffloadingHandler(
            model_name="m", tp_size=1, tp_rank=0, dtype=torch.float16,
            gpu_blocks_per_file=4, kv_caches=kv_caches,
            attn_backends=mock_attn_backends, bucket="b",
        )

        assert handler.wait(999) is True

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_wait_failed_job_returns_false(self, mock_client_cls, mock_stream, mock_attn_backends):
        from llmd_s3_backend.worker import S3GPUOffloadingHandler
        from concurrent.futures import Future

        kv_caches = make_kv_caches(num_layers=1, num_blocks=10)
        handler = S3GPUOffloadingHandler(
            model_name="m", tp_size=1, tp_rank=0, dtype=torch.float16,
            gpu_blocks_per_file=4, kv_caches=kv_caches,
            attn_backends=mock_attn_backends, bucket="b",
        )

        # Create a future that raises
        future = Future()
        future.set_exception(RuntimeError("boom"))
        handler.pending_futures[42] = future

        assert handler.wait(42) is False


class TestS3GPUGetFinished:
    """Test S3GPUOffloadingHandler.get_finished (overridden version)."""

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_returns_and_clears(self, mock_client_cls, mock_stream, mock_attn_backends):
        from llmd_s3_backend.worker import S3GPUOffloadingHandler

        kv_caches = make_kv_caches(num_layers=1, num_blocks=10)
        handler = S3GPUOffloadingHandler(
            model_name="m", tp_size=1, tp_rank=0, dtype=torch.float16,
            gpu_blocks_per_file=4, kv_caches=kv_caches,
            attn_backends=mock_attn_backends, bucket="b",
        )

        handler.completed_jobs = [(1, True), (2, False)]
        results = handler.get_finished()
        assert len(results) == 2
        assert handler.get_finished() == []


class TestS3GPUInitIoUring:
    """Test S3GPUOffloadingHandler io_uring initialization paths."""

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_iouring_disabled_by_default(self, mock_client_cls, mock_stream, mock_attn_backends):
        from llmd_s3_backend.worker import S3GPUOffloadingHandler

        kv_caches = make_kv_caches(num_layers=1, num_blocks=10)
        handler = S3GPUOffloadingHandler(
            model_name="m", tp_size=1, tp_rank=0, dtype=torch.float16,
            gpu_blocks_per_file=4, kv_caches=kv_caches,
            attn_backends=mock_attn_backends, bucket="b",
        )
        assert handler.enable_iouring is False
        assert handler.iouring_pool is None
        assert handler.pinned_buffer_pool is None

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    @patch("llmd_s3_backend.worker.IOURING_AVAILABLE", False)
    def test_iouring_fallback_when_unavailable(self, mock_client_cls, mock_stream, mock_attn_backends):
        from llmd_s3_backend.worker import S3GPUOffloadingHandler

        kv_caches = make_kv_caches(num_layers=1, num_blocks=10)
        handler = S3GPUOffloadingHandler(
            model_name="m", tp_size=1, tp_rank=0, dtype=torch.float16,
            gpu_blocks_per_file=4, kv_caches=kv_caches,
            attn_backends=mock_attn_backends, bucket="b",
            io_driver="io_uring",
        )
        assert handler.enable_iouring is False
        assert handler.io_driver == "crt"


class TestHandlerCleanup:
    """Test handler resource cleanup."""

    @patch("llmd_s3_backend.worker.torch.cuda.Stream", return_value=Mock())
    @patch("llmd_s3_backend.worker.S3ClientWrapper")
    def test_del_shuts_down_executor(self, mock_client_cls, mock_stream, mock_attn_backends):
        HandlerCls = _make_concrete_put_handler_cls()
        kv_caches = make_kv_caches(num_layers=1)
        handler = HandlerCls(
            model_name="m", tp_size=1, tp_rank=0,
            kv_caches=kv_caches, gpu_blocks_per_file=1,
            attn_backends=mock_attn_backends, dtype=torch.float16, bucket="b",
        )
        executor = handler.executor
        handler.__del__()
        assert executor._shutdown


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
