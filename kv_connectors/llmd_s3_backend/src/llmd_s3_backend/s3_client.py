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

import os
import boto3
from typing import Optional, Dict, Any
from botocore.config import Config
from vllm.logger import init_logger

logger = init_logger(__name__)


class S3ClientWrapper:
    """
    Wrapper for S3 client with credential management and configuration.
    Supports AWS environment variables, credentials file, and IAM roles.
    """

    def __init__(
        self,
        bucket: str,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        addressing_style: str = "auto",
        profile_name: Optional[str] = None,
    ):
        """
        Initialize S3 client wrapper.

        Args:
            bucket: S3 bucket name
            region: AWS region (defaults to env var or config)
            endpoint_url: Custom endpoint URL (for S3-compatible services)
            addressing_style: S3 addressing style (auto, path, virtual)
            profile_name: AWS profile name from credentials file
        """
        self.bucket = bucket
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.endpoint_url = endpoint_url
        self.addressing_style = addressing_style

        # Configure boto3 client
        config = Config(
            region_name=self.region,
            s3={"addressing_style": addressing_style},
            max_pool_connections=100,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )

        # Create session with optional profile
        session_kwargs: Dict[str, Any] = {}
        if profile_name:
            session_kwargs["profile_name"] = profile_name

        session = boto3.Session(**session_kwargs)

        # Create S3 client
        client_kwargs: Dict[str, Any] = {"config": config}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url

        self.s3_client = session.client("s3", **client_kwargs)

        logger.info(
            f"S3ClientWrapper initialized: bucket={bucket}, "
            f"region={self.region}, endpoint_url={endpoint_url}, "
            f"addressing_style={addressing_style}"
        )

    def object_exists(self, key: str) -> bool:
        """Check if an object exists in S3."""
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.s3_client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def put_object(self, key: str, data: bytes) -> None:
        """Upload data to S3."""
        self.s3_client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def get_object(self, key: str) -> bytes:
        """Download data from S3."""
        response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()
    
    def get_object_with_etag(self, key: str) -> tuple[bytes, str]:
        """
        Download data from S3 with ETag for conditional updates.
        
        Returns:
            Tuple of (data, etag)
        """
        response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        data = response["Body"].read()
        etag = response["ETag"].strip('"')  # Remove quotes from ETag
        return data, etag
    
    def put_object_if_match(self, key: str, data: bytes, etag: str) -> None:
        """
        Upload data to S3 only if ETag matches (conditional PUT).
        
        Args:
            key: S3 object key
            data: Data to upload
            etag: Expected ETag value
            
        Raises:
            botocore.exceptions.ClientError: If ETag doesn't match (412 Precondition Failed)
        """
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            IfMatch=etag
        )

    def delete_object(self, key: str) -> None:
        """Delete an object from S3."""
        self.s3_client.delete_object(Bucket=self.bucket, Key=key)

    def list_objects(self, prefix: str) -> list[str]:
        """List objects with a given prefix."""
        paginator = self.s3_client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            if "Contents" in page:
                keys.extend([obj["Key"] for obj in page["Contents"]])
        return keys

# Made with Bob
