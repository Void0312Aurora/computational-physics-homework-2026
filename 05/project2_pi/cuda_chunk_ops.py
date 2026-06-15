from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


PROJECT2_CUDA_CHUNK_OPS_NAME = "project2_cuda_chunk_ops_ext"
PROJECT2_CUDA_CHUNK_OPS_DIR = Path(__file__).resolve().parent.parent / "cuda"
PROJECT2_CUDA_CHUNK_OPS_SOURCES = [
    str(PROJECT2_CUDA_CHUNK_OPS_DIR / "project2_chunk_ops.cpp"),
    str(PROJECT2_CUDA_CHUNK_OPS_DIR / "project2_chunk_ops.cu"),
]


@lru_cache(maxsize=1)
def load_project2_cuda_chunk_ops():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot build project2 CUDA chunk ops")
    if os.environ.get("PROJECT2_DISABLE_CUDA_CHUNK_OPS_BUILD") == "1":
        raise RuntimeError("project2 CUDA chunk ops build disabled by environment")

    return load(
        name=PROJECT2_CUDA_CHUNK_OPS_NAME,
        sources=PROJECT2_CUDA_CHUNK_OPS_SOURCES,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def cuda_chunk_ops_available() -> bool:
    try:
        load_project2_cuda_chunk_ops()
    except Exception:
        return False
    return True


def trim_chunks_base65536(chunks: torch.Tensor) -> torch.Tensor:
    return load_project2_cuda_chunk_ops().trim_chunks_base65536(chunks)


def compare_abs_base65536(left: torch.Tensor, right: torch.Tensor) -> int:
    return int(load_project2_cuda_chunk_ops().compare_abs_base65536(left, right))


def add_abs_base65536(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return load_project2_cuda_chunk_ops().add_abs_base65536(left, right)


def sub_abs_base65536(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return load_project2_cuda_chunk_ops().sub_abs_base65536(left, right)


def mul_small_base65536(chunks: torch.Tensor, multiplier: int) -> torch.Tensor:
    return load_project2_cuda_chunk_ops().mul_small_base65536(chunks, int(multiplier))


def div2_base65536(chunks: torch.Tensor) -> torch.Tensor:
    return load_project2_cuda_chunk_ops().div2_base65536(chunks)
