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
AWS Signature Version 4 (SigV4) signing for S3 requests.

This module implements the AWS SigV4 signing algorithm for authenticating
S3 requests made via io_uring, bypassing the standard boto3/CRT clients.
"""

import hashlib
import hmac
import urllib.parse
from datetime import datetime
from typing import Dict, Optional
from vllm.logger import init_logger

logger = init_logger(__name__)


class S3SigV4Signer:
    """
    AWS Signature Version 4 signer for S3 requests.
    
    Implements the SigV4 signing algorithm as specified in:
    https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html
    """
    
    def __init__(
        self,
        access_key: str,
        secret_key: str,
        region: str = "us-east-1",
        service: str = "s3"
    ):
        """
        Initialize SigV4 signer.
        
        Args:
            access_key: AWS access key ID
            secret_key: AWS secret access key
            region: AWS region (default: us-east-1)
            service: AWS service name (default: s3)
        """
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.service = service
    
    def sign_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        payload: bytes = b"",
        timestamp: Optional[datetime] = None
    ) -> Dict[str, str]:
        """
        Sign an HTTP request using AWS SigV4.
        
        Args:
            method: HTTP method (GET, PUT, etc.)
            url: Full URL including scheme, host, and path
            headers: Optional HTTP headers
            payload: Request body (empty for GET)
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            Dictionary of headers including Authorization header
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Parse URL
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc
        path = parsed.path or "/"
        query = parsed.query
        
        # Initialize headers
        if headers is None:
            headers = {}
        headers = headers.copy()
        
        # Add required headers
        amz_date = timestamp.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = timestamp.strftime("%Y%m%d")
        
        headers["Host"] = host
        headers["x-amz-date"] = amz_date
        headers["x-amz-content-sha256"] = self._sha256_hash(payload)
        
        # Step 1: Create canonical request
        canonical_request = self._create_canonical_request(
            method, path, query, headers, payload
        )
        
        # Step 2: Create string to sign
        credential_scope = f"{date_stamp}/{self.region}/{self.service}/aws4_request"
        string_to_sign = self._create_string_to_sign(
            amz_date, credential_scope, canonical_request
        )
        
        # Step 3: Calculate signature
        signing_key = self._get_signature_key(
            self.secret_key, date_stamp, self.region, self.service
        )
        signature = hmac.new(
            signing_key,
            string_to_sign.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        # Step 4: Add authorization header
        signed_headers = ";".join(sorted(k.lower() for k in headers.keys()))
        authorization_header = (
            f"AWS4-HMAC-SHA256 "
            f"Credential={self.access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, "
            f"Signature={signature}"
        )
        
        headers["Authorization"] = authorization_header
        
        return headers
    
    def _create_canonical_request(
        self,
        method: str,
        path: str,
        query: str,
        headers: Dict[str, str],
        payload: bytes
    ) -> str:
        """Create canonical request string."""
        # Canonical URI
        canonical_uri = urllib.parse.quote(path, safe="/")
        
        # Canonical query string
        if query:
            query_params = urllib.parse.parse_qsl(query, keep_blank_values=True)
            query_params.sort()
            canonical_query = "&".join(
                f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(v, safe='')}"
                for k, v in query_params
            )
        else:
            canonical_query = ""
        
        # Canonical headers
        canonical_headers = ""
        for key in sorted(headers.keys()):
            canonical_headers += f"{key.lower()}:{headers[key].strip()}\n"
        
        # Signed headers
        signed_headers = ";".join(sorted(k.lower() for k in headers.keys()))
        
        # Payload hash
        payload_hash = self._sha256_hash(payload)
        
        # Combine into canonical request
        canonical_request = "\n".join([
            method.upper(),
            canonical_uri,
            canonical_query,
            canonical_headers,
            signed_headers,
            payload_hash
        ])
        
        return canonical_request
    
    def _create_string_to_sign(
        self,
        amz_date: str,
        credential_scope: str,
        canonical_request: str
    ) -> str:
        """Create string to sign."""
        canonical_request_hash = self._sha256_hash(canonical_request.encode("utf-8"))
        
        string_to_sign = "\n".join([
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            canonical_request_hash
        ])
        
        return string_to_sign
    
    def _get_signature_key(
        self,
        key: str,
        date_stamp: str,
        region: str,
        service: str
    ) -> bytes:
        """Derive signing key."""
        k_date = self._sign(f"AWS4{key}".encode("utf-8"), date_stamp)
        k_region = self._sign(k_date, region)
        k_service = self._sign(k_region, service)
        k_signing = self._sign(k_service, "aws4_request")
        return k_signing
    
    @staticmethod
    def _sign(key: bytes, msg: str) -> bytes:
        """HMAC-SHA256 signing."""
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()
    
    @staticmethod
    def _sha256_hash(data: bytes) -> str:
        """SHA256 hash as hex string."""
        return hashlib.sha256(data).hexdigest()


class S3RequestBuilder:
    """
    Helper class to build S3 HTTP requests with SigV4 signing.
    """
    
    def __init__(
        self,
        signer: S3SigV4Signer,
        endpoint: str,
        bucket: str
    ):
        """
        Initialize request builder.
        
        Args:
            signer: SigV4 signer instance
            endpoint: S3 endpoint URL (e.g., "https://s3.amazonaws.com")
            bucket: S3 bucket name
        """
        self.signer = signer
        self.endpoint = endpoint.rstrip("/")
        self.bucket = bucket
    
    def build_get_request(
        self,
        key: str,
        headers: Optional[Dict[str, str]] = None
    ) -> tuple[str, Dict[str, str]]:
        """
        Build a signed GET request for an S3 object.
        
        Args:
            key: S3 object key
            headers: Optional additional headers
            
        Returns:
            Tuple of (url, signed_headers)
        """
        url = f"{self.endpoint}/{self.bucket}/{key}"
        signed_headers = self.signer.sign_request(
            method="GET",
            url=url,
            headers=headers,
            payload=b""
        )
        
        return url, signed_headers
    
    def build_put_request(
        self,
        key: str,
        data: bytes,
        headers: Optional[Dict[str, str]] = None
    ) -> tuple[str, Dict[str, str]]:
        """
        Build a signed PUT request for an S3 object.
        
        Args:
            key: S3 object key
            data: Object data to upload
            headers: Optional additional headers
            
        Returns:
            Tuple of (url, signed_headers)
        """
        if headers is None:
            headers = {}
        
        # Add content-length header
        headers["Content-Length"] = str(len(data))
        
        url = f"{self.endpoint}/{self.bucket}/{key}"
        signed_headers = self.signer.sign_request(
            method="PUT",
            url=url,
            headers=headers,
            payload=data
        )
        
        return url, signed_headers
    
    def build_http_request_bytes(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: bytes = b""
    ) -> bytes:
        """
        Build complete HTTP request as bytes for socket transmission.
        
        Args:
            method: HTTP method
            url: Full URL
            headers: HTTP headers
            body: Request body
            
        Returns:
            Complete HTTP request as bytes
        """
        # Parse URL for path
        parsed = urllib.parse.urlparse(url)
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"
        
        # Build request line
        request_line = f"{method} {path} HTTP/1.1\r\n"
        
        # Build headers
        header_lines = ""
        for key, value in headers.items():
            header_lines += f"{key}: {value}\r\n"
        
        # Combine
        request = request_line + header_lines + "\r\n"
        request_bytes = request.encode("utf-8") + body
        
        return request_bytes


# Example usage and testing
if __name__ == "__main__":
    # Test SigV4 signing
    signer = S3SigV4Signer(
        access_key="AKIAIOSFODNN7EXAMPLE",
        secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        region="us-east-1"
    )
    
    # Test GET request
    url = "https://s3.amazonaws.com/examplebucket/test.txt"
    headers = signer.sign_request(
        method="GET",
        url=url,
        timestamp=datetime(2013, 5, 24, 0, 0, 0)
    )
    
    print("Signed GET request headers:")
    for key, value in headers.items():
        print(f"  {key}: {value}")
    
    # Test request builder
    builder = S3RequestBuilder(signer, "https://s3.amazonaws.com", "examplebucket")
    url, headers = builder.build_get_request("test.txt")
    
    print(f"\nBuilt GET request:")
    print(f"  URL: {url}")
    print(f"  Headers: {len(headers)} headers")
    
    # Build complete HTTP request
    request_bytes = builder.build_http_request_bytes("GET", url, headers)
    print(f"\nHTTP request size: {len(request_bytes)} bytes")
    print(f"First 200 bytes:\n{request_bytes[:200].decode('utf-8', errors='replace')}")

# Made with Bob
