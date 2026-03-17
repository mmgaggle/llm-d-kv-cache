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

"""Tests for AWS SigV4 signing and S3 request building."""

import hashlib
import hmac
import pytest
from datetime import datetime

from llmd_s3_backend.s3_auth import S3SigV4Signer, S3RequestBuilder

# AWS example credentials from public SigV4 documentation
# https://docs.aws.amazon.com/general/latest/gr/sigv4-calculate-signature.html
AWS_EXAMPLE_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
AWS_EXAMPLE_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
AWS_EXAMPLE_REGION = "us-east-1"
FIXED_TIMESTAMP = datetime(2024, 1, 15, 12, 30, 0)


@pytest.fixture
def signer():
    """Create a signer with example credentials."""
    return S3SigV4Signer(
        access_key=AWS_EXAMPLE_ACCESS_KEY,
        secret_key=AWS_EXAMPLE_SECRET_KEY,
        region=AWS_EXAMPLE_REGION,
    )


@pytest.fixture
def builder(signer):
    """Create a request builder with example credentials."""
    return S3RequestBuilder(
        signer=signer,
        endpoint="https://s3.amazonaws.com",
        bucket="examplebucket",
    )


# -------------------------------------------------------------------
# S3SigV4Signer
# -------------------------------------------------------------------
class TestS3SigV4SignerInit:
    """Test S3SigV4Signer initialization."""

    def test_stores_credentials(self):
        signer = S3SigV4Signer(
            access_key="AKID",
            secret_key="SECRET",
            region="eu-west-1",
            service="s3",
        )
        assert signer.access_key == "AKID"
        assert signer.secret_key == "SECRET"
        assert signer.region == "eu-west-1"
        assert signer.service == "s3"

    def test_defaults(self):
        signer = S3SigV4Signer(access_key="A", secret_key="B")
        assert signer.region == "us-east-1"
        assert signer.service == "s3"


class TestSha256Hash:
    """Test the _sha256_hash static method."""

    def test_empty_payload(self):
        expected = hashlib.sha256(b"").hexdigest()
        assert S3SigV4Signer._sha256_hash(b"") == expected

    def test_known_payload(self):
        data = b"hello world"
        expected = hashlib.sha256(data).hexdigest()
        assert S3SigV4Signer._sha256_hash(data) == expected

    def test_returns_lowercase_hex(self):
        result = S3SigV4Signer._sha256_hash(b"test")
        assert result == result.lower()
        assert len(result) == 64


class TestSign:
    """Test the _sign static method (HMAC-SHA256)."""

    def test_known_hmac(self):
        key = b"secret-key"
        msg = "test-message"
        expected = hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()
        assert S3SigV4Signer._sign(key, msg) == expected

    def test_different_keys_produce_different_signatures(self):
        sig1 = S3SigV4Signer._sign(b"key1", "msg")
        sig2 = S3SigV4Signer._sign(b"key2", "msg")
        assert sig1 != sig2

    def test_different_messages_produce_different_signatures(self):
        sig1 = S3SigV4Signer._sign(b"key", "msg1")
        sig2 = S3SigV4Signer._sign(b"key", "msg2")
        assert sig1 != sig2


class TestGetSignatureKey:
    """Test the signing key derivation chain."""

    def test_derivation_chain(self, signer):
        """Verify the 4-step HMAC chain: date → region → service → aws4_request."""
        date_stamp = "20240115"
        key = signer._get_signature_key(
            AWS_EXAMPLE_SECRET_KEY, date_stamp, AWS_EXAMPLE_REGION, "s3"
        )

        # Reproduce the chain manually
        k_date = hmac.new(
            f"AWS4{AWS_EXAMPLE_SECRET_KEY}".encode("utf-8"),
            date_stamp.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        k_region = hmac.new(
            k_date, AWS_EXAMPLE_REGION.encode("utf-8"), hashlib.sha256
        ).digest()
        k_service = hmac.new(
            k_region, b"s3", hashlib.sha256
        ).digest()
        k_signing = hmac.new(
            k_service, b"aws4_request", hashlib.sha256
        ).digest()

        assert key == k_signing

    def test_different_dates_produce_different_keys(self, signer):
        k1 = signer._get_signature_key(
            AWS_EXAMPLE_SECRET_KEY, "20240101", AWS_EXAMPLE_REGION, "s3"
        )
        k2 = signer._get_signature_key(
            AWS_EXAMPLE_SECRET_KEY, "20240102", AWS_EXAMPLE_REGION, "s3"
        )
        assert k1 != k2

    def test_different_regions_produce_different_keys(self, signer):
        k1 = signer._get_signature_key(
            AWS_EXAMPLE_SECRET_KEY, "20240115", "us-east-1", "s3"
        )
        k2 = signer._get_signature_key(
            AWS_EXAMPLE_SECRET_KEY, "20240115", "eu-west-1", "s3"
        )
        assert k1 != k2


class TestCreateCanonicalRequest:
    """Test canonical request string construction."""

    def test_get_request(self, signer):
        headers = {
            "Host": "s3.amazonaws.com",
            "x-amz-date": "20240115T123000Z",
            "x-amz-content-sha256": S3SigV4Signer._sha256_hash(b""),
        }
        cr = signer._create_canonical_request(
            method="GET",
            path="/examplebucket/test.txt",
            query="",
            headers=headers,
            payload=b"",
        )

        lines = cr.split("\n")
        assert lines[0] == "GET"
        assert lines[1] == "/examplebucket/test.txt"
        assert lines[2] == ""  # empty query string

    def test_method_is_uppercased(self, signer):
        headers = {"Host": "s3.amazonaws.com"}
        cr = signer._create_canonical_request("get", "/", "", headers, b"")
        assert cr.split("\n")[0] == "GET"

    def test_headers_sorted_lowercase(self, signer):
        headers = {
            "X-Amz-Date": "20240115T123000Z",
            "Host": "s3.amazonaws.com",
            "Content-Type": "application/octet-stream",
        }
        cr = signer._create_canonical_request("GET", "/", "", headers, b"")
        lines = cr.split("\n")

        # Find canonical headers section (after empty query string)
        # Headers are between the query string line and the signed headers line
        header_lines = []
        for line in lines[3:]:
            if ":" in line:
                header_lines.append(line)
            else:
                break

        header_names = [h.split(":")[0] for h in header_lines]
        assert header_names == sorted(header_names)

    def test_query_string_sorted(self, signer):
        headers = {"Host": "s3.amazonaws.com"}
        cr = signer._create_canonical_request(
            "GET", "/", "z=1&a=2&m=3", headers, b""
        )
        lines = cr.split("\n")
        # Query string is line[2]
        assert lines[2] == "a=2&m=3&z=1"

    def test_empty_path_defaults_to_slash(self, signer):
        headers = {"Host": "s3.amazonaws.com"}
        # The signer handles empty path in sign_request, but _create_canonical_request
        # gets the already-parsed path; verify "/" works
        cr = signer._create_canonical_request("GET", "/", "", headers, b"")
        lines = cr.split("\n")
        assert lines[1] == "/"

    def test_payload_hash_included(self, signer):
        payload = b"test-body-data"
        headers = {"Host": "s3.amazonaws.com"}
        cr = signer._create_canonical_request("PUT", "/key", "", headers, payload)
        lines = cr.split("\n")
        # Last line is the payload hash
        assert lines[-1] == S3SigV4Signer._sha256_hash(payload)

    def test_signed_headers_list(self, signer):
        headers = {
            "Host": "s3.amazonaws.com",
            "X-Amz-Date": "20240115T123000Z",
        }
        cr = signer._create_canonical_request("GET", "/", "", headers, b"")
        lines = cr.split("\n")
        # Signed headers is the second-to-last line
        assert lines[-2] == "host;x-amz-date"


class TestCreateStringToSign:
    """Test string-to-sign construction."""

    def test_format(self, signer):
        amz_date = "20240115T123000Z"
        scope = "20240115/us-east-1/s3/aws4_request"
        canonical_request = "GET\n/\n\nhost:s3.amazonaws.com\n\nhost\nhash"

        sts = signer._create_string_to_sign(amz_date, scope, canonical_request)
        lines = sts.split("\n")

        assert lines[0] == "AWS4-HMAC-SHA256"
        assert lines[1] == amz_date
        assert lines[2] == scope
        assert lines[3] == S3SigV4Signer._sha256_hash(
            canonical_request.encode("utf-8")
        )


class TestSignRequest:
    """Test the full sign_request flow."""

    def test_returns_authorization_header(self, signer):
        headers = signer.sign_request(
            method="GET",
            url="https://s3.amazonaws.com/examplebucket/test.txt",
            timestamp=FIXED_TIMESTAMP,
        )
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")

    def test_authorization_format(self, signer):
        headers = signer.sign_request(
            method="GET",
            url="https://s3.amazonaws.com/examplebucket/test.txt",
            timestamp=FIXED_TIMESTAMP,
        )
        auth = headers["Authorization"]

        assert "Credential=" in auth
        assert f"Credential={AWS_EXAMPLE_ACCESS_KEY}/" in auth
        assert "SignedHeaders=" in auth
        assert "Signature=" in auth

    def test_required_headers_added(self, signer):
        headers = signer.sign_request(
            method="GET",
            url="https://s3.amazonaws.com/examplebucket/test.txt",
            timestamp=FIXED_TIMESTAMP,
        )
        assert headers["Host"] == "s3.amazonaws.com"
        assert headers["x-amz-date"] == "20240115T123000Z"
        assert "x-amz-content-sha256" in headers

    def test_empty_payload_hash(self, signer):
        headers = signer.sign_request(
            method="GET",
            url="https://s3.amazonaws.com/examplebucket/test.txt",
            timestamp=FIXED_TIMESTAMP,
        )
        assert headers["x-amz-content-sha256"] == S3SigV4Signer._sha256_hash(b"")

    def test_payload_hash_matches_body(self, signer):
        payload = b"upload-data"
        headers = signer.sign_request(
            method="PUT",
            url="https://s3.amazonaws.com/examplebucket/test.txt",
            headers=None,
            payload=payload,
            timestamp=FIXED_TIMESTAMP,
        )
        assert headers["x-amz-content-sha256"] == S3SigV4Signer._sha256_hash(payload)

    def test_existing_headers_preserved(self, signer):
        custom_headers = {"Content-Type": "application/octet-stream"}
        headers = signer.sign_request(
            method="PUT",
            url="https://s3.amazonaws.com/examplebucket/test.txt",
            headers=custom_headers,
            payload=b"data",
            timestamp=FIXED_TIMESTAMP,
        )
        assert headers["Content-Type"] == "application/octet-stream"
        # Original dict should not be mutated
        assert "Authorization" not in custom_headers

    def test_credential_scope_format(self, signer):
        headers = signer.sign_request(
            method="GET",
            url="https://s3.amazonaws.com/examplebucket/test.txt",
            timestamp=FIXED_TIMESTAMP,
        )
        auth = headers["Authorization"]
        # Credential=AKIAIOSFODNN7EXAMPLE/20240115/us-east-1/s3/aws4_request
        expected_scope = "20240115/us-east-1/s3/aws4_request"
        assert expected_scope in auth

    def test_deterministic_signatures(self, signer):
        """Same inputs must produce the same signature."""
        h1 = signer.sign_request(
            method="GET",
            url="https://s3.amazonaws.com/examplebucket/test.txt",
            timestamp=FIXED_TIMESTAMP,
        )
        h2 = signer.sign_request(
            method="GET",
            url="https://s3.amazonaws.com/examplebucket/test.txt",
            timestamp=FIXED_TIMESTAMP,
        )
        assert h1["Authorization"] == h2["Authorization"]

    def test_different_methods_produce_different_signatures(self, signer):
        url = "https://s3.amazonaws.com/examplebucket/test.txt"
        h_get = signer.sign_request(method="GET", url=url, timestamp=FIXED_TIMESTAMP)
        h_put = signer.sign_request(
            method="PUT", url=url, payload=b"data", timestamp=FIXED_TIMESTAMP
        )
        assert h_get["Authorization"] != h_put["Authorization"]

    def test_different_paths_produce_different_signatures(self, signer):
        ts = FIXED_TIMESTAMP
        h1 = signer.sign_request(
            method="GET",
            url="https://s3.amazonaws.com/bucket/file1.txt",
            timestamp=ts,
        )
        h2 = signer.sign_request(
            method="GET",
            url="https://s3.amazonaws.com/bucket/file2.txt",
            timestamp=ts,
        )
        assert h1["Authorization"] != h2["Authorization"]

    def test_url_with_query_string(self, signer):
        headers = signer.sign_request(
            method="GET",
            url="https://s3.amazonaws.com/bucket/key?list-type=2&prefix=foo",
            timestamp=FIXED_TIMESTAMP,
        )
        assert "Authorization" in headers

    def test_signed_headers_list_sorted(self, signer):
        headers = signer.sign_request(
            method="GET",
            url="https://s3.amazonaws.com/bucket/key",
            headers={"X-Custom": "val"},
            timestamp=FIXED_TIMESTAMP,
        )
        auth = headers["Authorization"]
        # Extract SignedHeaders= value
        sh_start = auth.index("SignedHeaders=") + len("SignedHeaders=")
        sh_end = auth.index(",", sh_start)
        signed = auth[sh_start:sh_end].split(";")
        assert signed == sorted(signed)

    def test_uses_current_time_when_no_timestamp(self, signer):
        """When timestamp is None, headers should use current UTC time."""
        headers = signer.sign_request(
            method="GET",
            url="https://s3.amazonaws.com/bucket/key",
        )
        assert "x-amz-date" in headers
        # Should be a valid timestamp format: YYYYMMDDTHHMMSSZ
        amz_date = headers["x-amz-date"]
        assert len(amz_date) == 16
        assert amz_date.endswith("Z")


# -------------------------------------------------------------------
# S3RequestBuilder
# -------------------------------------------------------------------
class TestS3RequestBuilderInit:
    """Test S3RequestBuilder initialization."""

    def test_stores_attributes(self, signer):
        builder = S3RequestBuilder(
            signer=signer,
            endpoint="https://s3.us-west-2.amazonaws.com",
            bucket="my-bucket",
        )
        assert builder.signer is signer
        assert builder.endpoint == "https://s3.us-west-2.amazonaws.com"
        assert builder.bucket == "my-bucket"

    def test_trailing_slash_stripped(self, signer):
        builder = S3RequestBuilder(
            signer=signer,
            endpoint="https://s3.amazonaws.com/",
            bucket="bucket",
        )
        assert builder.endpoint == "https://s3.amazonaws.com"


class TestBuildGetRequest:
    """Test building signed GET requests."""

    def test_returns_url_and_headers(self, builder):
        url, headers = builder.build_get_request("path/to/key.bin")
        assert url == "https://s3.amazonaws.com/examplebucket/path/to/key.bin"
        assert isinstance(headers, dict)
        assert "Authorization" in headers

    def test_get_uses_empty_payload(self, builder):
        _, headers = builder.build_get_request("key")
        assert headers["x-amz-content-sha256"] == S3SigV4Signer._sha256_hash(b"")

    def test_additional_headers_passed_through(self, builder):
        _, headers = builder.build_get_request(
            "key", headers={"Range": "bytes=0-1023"}
        )
        assert headers["Range"] == "bytes=0-1023"


class TestBuildPutRequest:
    """Test building signed PUT requests."""

    def test_returns_url_and_headers(self, builder):
        data = b"file-contents"
        url, headers = builder.build_put_request("path/to/key.bin", data)
        assert url == "https://s3.amazonaws.com/examplebucket/path/to/key.bin"
        assert "Authorization" in headers

    def test_content_length_header_added(self, builder):
        data = b"hello"
        _, headers = builder.build_put_request("key", data)
        assert headers["Content-Length"] == str(len(data))

    def test_payload_hash_matches_data(self, builder):
        data = b"payload-bytes"
        _, headers = builder.build_put_request("key", data)
        assert headers["x-amz-content-sha256"] == S3SigV4Signer._sha256_hash(data)

    def test_empty_data(self, builder):
        _, headers = builder.build_put_request("key", b"")
        assert headers["Content-Length"] == "0"

    def test_additional_headers_merged(self, builder):
        _, headers = builder.build_put_request(
            "key",
            b"data",
            headers={"Content-Type": "application/octet-stream"},
        )
        assert headers["Content-Type"] == "application/octet-stream"
        assert "Content-Length" in headers


class TestBuildHttpRequestBytes:
    """Test building raw HTTP request bytes."""

    def test_request_line(self, builder):
        url = "https://s3.amazonaws.com/examplebucket/key.bin"
        headers = {"Host": "s3.amazonaws.com", "x-amz-date": "20240115T123000Z"}
        raw = builder.build_http_request_bytes("GET", url, headers)

        lines = raw.decode("utf-8").split("\r\n")
        assert lines[0] == "GET /examplebucket/key.bin HTTP/1.1"

    def test_headers_included(self, builder):
        url = "https://s3.amazonaws.com/bucket/key"
        headers = {"Host": "s3.amazonaws.com", "X-Custom": "value"}
        raw = builder.build_http_request_bytes("GET", url, headers)
        text = raw.decode("utf-8")

        assert "Host: s3.amazonaws.com\r\n" in text
        assert "X-Custom: value\r\n" in text

    def test_blank_line_separates_headers_from_body(self, builder):
        url = "https://s3.amazonaws.com/bucket/key"
        headers = {"Host": "s3.amazonaws.com"}
        raw = builder.build_http_request_bytes("GET", url, headers)
        text = raw.decode("utf-8")

        assert "\r\n\r\n" in text

    def test_body_appended(self, builder):
        url = "https://s3.amazonaws.com/bucket/key"
        headers = {"Host": "s3.amazonaws.com"}
        body = b"binary-body-data"
        raw = builder.build_http_request_bytes("PUT", url, headers, body=body)

        assert raw.endswith(body)

    def test_query_string_in_path(self, builder):
        url = "https://s3.amazonaws.com/bucket/key?uploadId=123"
        headers = {"Host": "s3.amazonaws.com"}
        raw = builder.build_http_request_bytes("GET", url, headers)
        text = raw.decode("utf-8")

        assert text.startswith("GET /bucket/key?uploadId=123 HTTP/1.1\r\n")

    def test_put_with_full_signing(self, builder):
        """End-to-end: build a signed PUT and serialize to bytes."""
        data = b"upload-payload"
        url, signed_headers = builder.build_put_request("test/key.bin", data)
        raw = builder.build_http_request_bytes("PUT", url, signed_headers, body=data)

        text = raw[: raw.index(b"\r\n\r\n")].decode("utf-8")
        assert "PUT" in text
        assert "Authorization:" in text
        assert raw.endswith(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
