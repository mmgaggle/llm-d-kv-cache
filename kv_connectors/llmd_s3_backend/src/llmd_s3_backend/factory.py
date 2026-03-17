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
S3 offloading connector factory registration.

This module registers :class:`~llmd_s3_backend.spec.S3OffloadingSpec`
with vLLM's :class:`OffloadingSpecFactory` so that it can be selected
at runtime via the ``kv_connector`` configuration key.

Registration happens at import time.  When the ``llmd_s3_backend``
package is installed and the user sets
``kv_connector = "S3OffloadingSpec"`` in their vLLM config, the
factory lazily imports :mod:`llmd_s3_backend.spec` and instantiates
the spec, which in turn creates the S3 manager and GPU↔S3 transfer
handlers.
"""

from vllm.logger import init_logger
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

logger = init_logger(__name__)

OffloadingSpecFactory.register_spec(
    "S3OffloadingSpec", "llmd_s3_backend.spec", "S3OffloadingSpec"
)
