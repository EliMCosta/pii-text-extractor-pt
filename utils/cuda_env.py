"""
CUDA environment sanitization utilities.

This module provides functions to avoid conflicts between system CUDA libraries
and the CUDA libraries bundled with PyTorch wheels.
"""

from __future__ import annotations

import os
import sys


def maybe_reexec_without_system_cuda() -> None:
    """
    Avoid picking up system CUDA libs via LD_LIBRARY_PATH.

    PyTorch CUDA wheels ship their own CUDA/cuBLAS/cuDNN libraries. If the user has
    a system CUDA toolkit (e.g. /usr/local/cuda) in LD_LIBRARY_PATH, the dynamic
    loader can pick those up instead, causing hard-to-debug GEMM failures
    (e.g. CUBLAS_STATUS_INVALID_VALUE for fp16/bf16).

    This function checks if system CUDA paths are present and, if so, re-executes
    the current process with a sanitized environment.
    """
    if os.environ.get("PII_TEXT_IDENTIFIER_CUDA_ENV_SANITIZED") == "1":
        return

    ld = os.environ.get("LD_LIBRARY_PATH")
    if not ld:
        return

    bad_prefixes = ("/usr/local/cuda/lib64", "/usr/local/cuda/lib")
    parts = [p for p in ld.split(":") if p]
    filtered = [p for p in parts if not any(p.startswith(bp) for bp in bad_prefixes)]
    if filtered != parts:
        # IMPORTANT: LD_LIBRARY_PATH is read by the dynamic loader at process start.
        # Changing it in-process may not affect subsequent dlopen() calls reliably.
        # So we re-exec the process once with a sanitized environment.
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = ":".join(filtered)
        env["PII_TEXT_IDENTIFIER_CUDA_ENV_SANITIZED"] = "1"
        # If the user set CUDA_HOME to the system toolkit, unset it to discourage
        # accidental linkage against the toolkit when CUDA wheels are used.
        if env.get("CUDA_HOME") == "/usr/local/cuda":
            env.pop("CUDA_HOME", None)

        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def maybe_init_distributed_with_device_id() -> None:
    """
    Initialize torch.distributed early with an explicit device_id.

    This removes the warning emitted by torch.distributed.barrier() when it falls
    back to "the device under current context" (common in multi-GPU runs launched
    via torchrun/accelerate when the script itself doesn't call init_process_group).
    """
    import torch

    if not torch.distributed.is_available():
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return

    if torch.distributed.is_initialized():
        return

    device_id: torch.device | int | None = None
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        cuda_devices = torch.cuda.device_count()
        if local_rank < 0 or local_rank >= cuda_devices:
            raise RuntimeError(
                f"Invalid LOCAL_RANK={local_rank} for CUDA device_count={cuda_devices}. "
                "Launch with one process per visible GPU (e.g. torchrun/accelerate)."
            )
        torch.cuda.set_device(local_rank)
        device_id = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    torch.distributed.init_process_group(
        backend=backend,
        init_method="env://",
        device_id=device_id,
    )
