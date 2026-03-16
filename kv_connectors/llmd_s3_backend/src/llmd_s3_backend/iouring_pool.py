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

"""
io_uring-based connection pool for zero-copy S3 transfers.

This module provides a high-performance connection pool that uses Linux io_uring
for zero-copy network I/O directly into pinned memory buffers.

NOTE: This is a prototype implementation. Full io_uring support requires:
- Linux kernel 5.1+
- liburing or python-liburing bindings
- Proper HTTP/2 framing and connection management
"""

import socket
import threading
import queue
import time
from typing import Optional, Dict, List
from dataclasses import dataclass
from vllm.logger import init_logger

from llmd_s3_backend.pinned_buffers import PinnedBuffer, PinnedBufferPool
from llmd_s3_backend.s3_auth import S3SigV4Signer, S3RequestBuilder

logger = init_logger(__name__)


@dataclass
class IoUringConfig:
    """Configuration for io_uring pool."""
    queue_depth: int = 1024
    num_workers: int = 16
    connection_timeout: int = 30
    read_timeout: int = 60
    max_connections_per_endpoint: int = 8


class HTTPConnection:
    """
    Represents a persistent HTTP/1.1 or HTTP/2 connection.
    
    In production, this would use HTTP/2 multiplexing and io_uring for I/O.
    This prototype uses standard sockets for compatibility.
    """
    
    def __init__(self, endpoint: str, timeout: int = 30):
        """
        Initialize HTTP connection.
        
        Args:
            endpoint: S3 endpoint URL (e.g., "s3.amazonaws.com:443")
            timeout: Connection timeout in seconds
        """
        self.endpoint = endpoint
        self.timeout = timeout
        self.socket: Optional[socket.socket] = None
        self.last_used = time.time()
        self.in_use = False
        self.request_count = 0
        
    def connect(self):
        """Establish connection to endpoint."""
        if self.socket is not None:
            return
        
        # Parse endpoint
        if ":" in self.endpoint:
            host, port = self.endpoint.rsplit(":", 1)
            port = int(port)
        else:
            host = self.endpoint
            port = 443  # Default HTTPS
        
        # Create socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(self.timeout)
        
        try:
            self.socket.connect((host, port))
            logger.debug(f"Connected to {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to {host}:{port}: {e}")
            self.socket = None
            raise
    
    def send_request(self, request_bytes: bytes) -> int:
        """
        Send HTTP request.
        
        Args:
            request_bytes: Complete HTTP request as bytes
            
        Returns:
            Number of bytes sent
        """
        if self.socket is None:
            raise RuntimeError("Not connected")
        
        total_sent = 0
        while total_sent < len(request_bytes):
            sent = self.socket.send(request_bytes[total_sent:])
            if sent == 0:
                raise RuntimeError("Socket connection broken")
            total_sent += sent
        
        self.request_count += 1
        self.last_used = time.time()
        return total_sent
    
    def recv_into_buffer(self, buffer: PinnedBuffer, size: int) -> int:
        """
        Receive data directly into pinned buffer.
        
        In production with io_uring, this would be a true zero-copy operation.
        This prototype copies data but demonstrates the API.
        
        Args:
            buffer: Pinned buffer to receive into
            size: Number of bytes to receive
            
        Returns:
            Number of bytes received
        """
        if self.socket is None:
            raise RuntimeError("Not connected")
        
        # Read HTTP response headers first
        headers = b""
        while b"\r\n\r\n" not in headers:
            chunk = self.socket.recv(1)
            if not chunk:
                raise RuntimeError("Connection closed while reading headers")
            headers += chunk
        
        # Parse content length from headers
        headers_str = headers.decode("utf-8", errors="ignore")
        content_length = 0
        for line in headers_str.split("\r\n"):
            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":", 1)[1].strip())
                break
        
        if content_length == 0:
            logger.warning("No Content-Length header found")
            return 0
        
        # Read body into buffer
        bytes_read = 0
        buffer_view = memoryview(buffer.tensor.numpy())
        
        while bytes_read < content_length and bytes_read < size:
            chunk_size = min(8192, content_length - bytes_read, size - bytes_read)
            chunk = self.socket.recv(chunk_size)
            if not chunk:
                break
            
            buffer_view[bytes_read:bytes_read + len(chunk)] = chunk
            bytes_read += len(chunk)
        
        self.last_used = time.time()
        return bytes_read
    
    def close(self):
        """Close connection."""
        if self.socket:
            try:
                self.socket.close()
            except Exception as e:
                logger.warning(f"Error closing socket: {e}")
            finally:
                self.socket = None
    
    def is_alive(self) -> bool:
        """Check if connection is still alive."""
        if self.socket is None:
            return False
        
        # Simple check - in production would use TCP keepalive
        age = time.time() - self.last_used
        return age < 300  # 5 minute idle timeout
    
    def __del__(self):
        """Cleanup on destruction."""
        self.close()


class IoUringPool:
    """
    Connection pool using io_uring for zero-copy S3 transfers.
    
    This is a prototype that demonstrates the architecture. Production
    implementation would use actual io_uring bindings for true zero-copy.
    """
    
    def __init__(
        self,
        signer: S3SigV4Signer,
        endpoints: List[str],
        bucket: str,
        buffer_pool: PinnedBufferPool,
        config: Optional[IoUringConfig] = None
    ):
        """
        Initialize io_uring pool with multipathing support.
        
        Args:
            signer: S3 SigV4 signer
            endpoints: List of S3 endpoints (e.g., ["10.0.1.5:443", "10.0.1.6:443"])
            bucket: S3 bucket name
            buffer_pool: Pinned buffer pool for zero-copy
            config: Optional configuration
        """
        self.signer = signer
        self.endpoints = endpoints if isinstance(endpoints, list) else [endpoints]
        self.bucket = bucket
        self.buffer_pool = buffer_pool
        self.config = config or IoUringConfig()
        
        # Connection pool per endpoint
        self.connections: Dict[str, List[HTTPConnection]] = {}
        self.connection_lock = threading.Lock()
        
        # Round-robin endpoint selection
        self.current_endpoint_idx = 0
        self.endpoint_lock = threading.Lock()
        
        # Request builder (use first endpoint for base URL)
        base_endpoint = self.endpoints[0]
        self.request_builder = S3RequestBuilder(signer, f"https://{base_endpoint}", bucket)
        
        # Statistics
        self.stats = {
            "requests": 0,
            "bytes_transferred": 0,
            "errors": 0,
            "cache_hits": 0,
        }
        self.stats_lock = threading.Lock()
        
        logger.info(
            f"IoUringPool initialized: endpoints={self.endpoints}, "
            f"bucket={bucket}, queue_depth={self.config.queue_depth}"
        )
    
    def _select_endpoint(self) -> str:
        """
        Select endpoint using round-robin load balancing.
        
        Returns:
            Selected endpoint string
        """
        with self.endpoint_lock:
            endpoint = self.endpoints[self.current_endpoint_idx]
            self.current_endpoint_idx = (self.current_endpoint_idx + 1) % len(self.endpoints)
            return endpoint
    
    def _get_connection(self, endpoint: str) -> HTTPConnection:
        """
        Get or create a connection to endpoint.
        
        Args:
            endpoint: Endpoint to connect to
            
        Returns:
            HTTPConnection instance
        """
        with self.connection_lock:
            # Get or create connection list for endpoint
            if endpoint not in self.connections:
                self.connections[endpoint] = []
            
            conn_list = self.connections[endpoint]
            
            # Find available connection
            for conn in conn_list:
                if not conn.in_use and conn.is_alive():
                    conn.in_use = True
                    return conn
            
            # Create new connection if under limit
            if len(conn_list) < self.config.max_connections_per_endpoint:
                conn = HTTPConnection(endpoint, self.config.connection_timeout)
                conn.in_use = True
                conn_list.append(conn)
                return conn
            
            # Wait for available connection (simplified - should use condition variable)
            logger.warning(f"Connection pool exhausted for {endpoint}")
            raise RuntimeError("Connection pool exhausted")
    
    def _release_connection(self, conn: HTTPConnection):
        """Release connection back to pool."""
        with self.connection_lock:
            conn.in_use = False
    
    def get_object_zerocopy(
        self,
        key: str,
        pinned_buffer: PinnedBuffer,
        timeout: Optional[float] = None
    ) -> int:
        """
        Get S3 object with zero-copy into pinned buffer.
        
        This is the core zero-copy operation. In production with io_uring:
        1. Submit io_uring read operation with pinned buffer
        2. Kernel DMAs data directly from NIC to pinned memory
        3. No intermediate copies
        
        Args:
            key: S3 object key
            pinned_buffer: Pre-allocated pinned buffer
            timeout: Optional timeout
            
        Returns:
            Number of bytes read
        """
        start_time = time.time()
        
        try:
            # Build signed GET request
            url, headers = self.request_builder.build_get_request(key)
            request_bytes = self.request_builder.build_http_request_bytes(
                "GET", url, headers
            )
            
            # Select endpoint and get connection
            endpoint = self._select_endpoint()
            conn = self._get_connection(endpoint)
            
            try:
                # Ensure connected
                if conn.socket is None:
                    conn.connect()
                
                # Send request
                conn.send_request(request_bytes)
                
                # Receive into pinned buffer (zero-copy with io_uring)
                bytes_read = conn.recv_into_buffer(pinned_buffer, pinned_buffer.size_bytes)
                
                # Update stats
                with self.stats_lock:
                    self.stats["requests"] += 1
                    self.stats["bytes_transferred"] += bytes_read
                
                elapsed = time.time() - start_time
                logger.debug(
                    f"GET {key}: {bytes_read} bytes in {elapsed:.3f}s "
                    f"({bytes_read / elapsed / 1024 / 1024:.1f} MB/s)"
                )
                
                return bytes_read
                
            finally:
                self._release_connection(conn)
                
        except Exception as e:
            with self.stats_lock:
                self.stats["errors"] += 1
            logger.error(f"Failed to GET {key}: {e}")
            raise
    
    def put_object_zerocopy(
        self,
        key: str,
        pinned_buffer: PinnedBuffer,
        size: int,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Put S3 object with zero-copy from pinned buffer.
        
        Args:
            key: S3 object key
            pinned_buffer: Pinned buffer containing data
            size: Number of bytes to write
            timeout: Optional timeout
            
        Returns:
            True if successful
        """
        start_time = time.time()
        
        try:
            # Get data from buffer
            data = bytes(pinned_buffer.tensor[:size].numpy())
            
            # Build signed PUT request
            url, headers = self.request_builder.build_put_request(key, data)
            request_bytes = self.request_builder.build_http_request_bytes(
                "PUT", url, headers, data
            )
            
            # Select endpoint and get connection
            endpoint = self._select_endpoint()
            conn = self._get_connection(endpoint)
            
            try:
                # Ensure connected
                if conn.socket is None:
                    conn.connect()
                
                # Send request (with body)
                conn.send_request(request_bytes)
                
                # Read response
                response = conn.socket.recv(4096)
                
                # Check for 200 OK
                success = b"200 OK" in response or b"204 No Content" in response
                
                # Update stats
                with self.stats_lock:
                    self.stats["requests"] += 1
                    if success:
                        self.stats["bytes_transferred"] += size
                    else:
                        self.stats["errors"] += 1
                
                elapsed = time.time() - start_time
                logger.debug(
                    f"PUT {key}: {size} bytes in {elapsed:.3f}s "
                    f"({size / elapsed / 1024 / 1024:.1f} MB/s)"
                )
                
                return success
                
            finally:
                self._release_connection(conn)
                
        except Exception as e:
            with self.stats_lock:
                self.stats["errors"] += 1
            logger.error(f"Failed to PUT {key}: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get pool statistics."""
        with self.stats_lock:
            stats = self.stats.copy()
        
        with self.connection_lock:
            total_connections = sum(len(conns) for conns in self.connections.values())
            active_connections = sum(
                sum(1 for c in conns if c.in_use)
                for conns in self.connections.values()
            )
        
        stats["total_connections"] = total_connections
        stats["active_connections"] = active_connections
        stats["endpoints"] = len(self.connections)
        
        return stats
    
    def close(self):
        """Close all connections."""
        with self.connection_lock:
            for conn_list in self.connections.values():
                for conn in conn_list:
                    conn.close()
            self.connections.clear()
        
        if logger:
            logger.info("IoUringPool closed")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup


# Example usage
if __name__ == "__main__":
    from llmd_s3_backend.pinned_buffers import PinnedBufferPool
    from llmd_s3_backend.s3_auth import S3SigV4Signer
    
    # Create components
    signer = S3SigV4Signer(
        access_key="test_key",
        secret_key="test_secret",
        region="us-east-1"
    )
    
    buffer_pool = PinnedBufferPool(buffer_size_mb=64, num_buffers=4)
    
    pool = IoUringPool(
        signer=signer,
        endpoints=["s3.amazonaws.com"],
        bucket="test-bucket",
        buffer_pool=buffer_pool
    )
    
    print("IoUringPool created successfully")
    print(f"Stats: {pool.get_stats()}")
    
    pool.close()

# Made with Bob
