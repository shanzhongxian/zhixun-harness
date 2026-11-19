"""批次协议适配层（T02）。

只做三件事：把 inbox 里的批次读成项目统一的数据结构、把结果写成 submit 产物、
按批次循环驱动 harness。协议细节见 docs/平台协议.md。
"""

from .batch import BatchRef, iter_batches, read_batch  # noqa: F401
from .artifact import atomic_write_json, sha256_file, write_artifact, write_no_op  # noqa: F401
