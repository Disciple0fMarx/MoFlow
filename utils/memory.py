"""Memory / resource safeguards shared by the SDD trainers and evidence tools.

SDD evidence runs (Q1–Q6, the 8×8 transfer sweep) are long-lived, batch-heavy
and were the source of a hard OOM system freeze. Three failure modes are
handled here:

1. **DataLoader workers multiply RAM.** Every worker is a forked process that
   carries its own copy of the dataset (annotation index, window tensors) and
   its own feature-lookup mapping. :func:`resolve_num_workers` clamps the count
   to a safe range and defaults to ``0`` (in-process loading, the SDD-correct
   choice), so no experiment can silently fan out RAM.
2. **Unreferenced tensors linger.** :func:`free_memory` is the explicit
   collect/empty-cache pair to call at phase boundaries (scene switch, every N
   eval batches, end of a sweep cell).
3. **A single oversized batch kills a whole sweep.** :func:`run_with_oom_split`
   retries a failing batch in halves so one cell with an unlucky batch size
   cannot abort 128 others.
"""

from __future__ import annotations

import gc
import logging
from typing import Callable, Iterable, Sequence, TypeVar

import numpy as np
import torch

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Upper bound for DataLoader workers. Beyond this the forked dataset copies
#: dominate RAM on the lab box (single 4080, 32 GB) with no throughput gain —
#: the SDD bottleneck is the GPU forward pass, not data loading.
MAX_NUM_WORKERS = 4


def resolve_num_workers(requested: int | None, default: int = 0) -> int:
    """Clamp a requested DataLoader worker count into ``[0, MAX_NUM_WORKERS]``.

    ``None``/negative values fall back to ``default``. Values above the cap are
    clamped (not rejected) so an over-eager config cannot OOM the host.
    """
    if requested is None:
        value = default
    else:
        value = int(requested)
    if value < 0:
        value = default
    return min(value, MAX_NUM_WORKERS)


def free_memory(collect: bool = True) -> None:
    """Release unreferenced Python objects and cached CUDA blocks."""
    if collect:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom(exc: BaseException) -> bool:
    """True for CUDA OOM and for host-side allocation failures."""
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text or "cannot allocate" in text


def _split(seq: Sequence[T]) -> tuple[Sequence[T], Sequence[T]]:
    mid = len(seq) // 2
    return seq[:mid], seq[mid:]


def run_with_oom_split(
    fn: Callable[[Sequence[T]], T],
    items: Sequence[T],
    *,
    min_chunk: int = 1,
    collect: bool = True,
    label: str = "batch",
) -> T:
    """Run ``fn`` over ``items``, halving the input on allocation failure.

    ``fn`` receives a sub-sequence and must return a value that the caller
    concatenates in order. On an OOM the chunk is split in two and retried, so a
    batch that does not fit is processed in smaller pieces rather than aborting
    the run. Returns the concatenation of the per-chunk results.
    """
    if not items:
        raise ValueError("run_with_oom_split called with an empty sequence")
    try:
        return fn(items)
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is an OOM
        if not is_oom(exc):
            raise
        if len(items) <= max(1, min_chunk):
            logger.error(
                "%s: OOM persists at the minimum chunk size (%d) — giving up",
                label,
                len(items),
            )
            raise
        half = max(1, len(items) // 2)
        logger.warning(
            "%s: OOM at size %d — retrying in halves (minimum chunk %d)",
            label,
            len(items),
            max(1, min_chunk),
        )
        free_memory(collect=collect)
        left, right = _split(items)
        out_l = run_with_oom_split(
            fn, left, min_chunk=min_chunk, collect=collect, label=label
        )
        free_memory(collect=collect)
        out_r = run_with_oom_split(
            fn, right, min_chunk=min_chunk, collect=collect, label=label
        )
        return _concat(out_l, out_r)


def _concat(left: T, right: T) -> T:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.cat([left, right], dim=0)  # type: ignore[return-value]
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.concatenate([left, right], axis=0)  # type: ignore[return-value]
    if isinstance(left, dict) and isinstance(right, dict):
        return {  # type: ignore[return-value]
            k: _concat(left[k], right[k]) for k in left.keys() | right.keys()
        }
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return type(left)(list(left) + list(right))  # type: ignore[return-value]
    raise TypeError(
        f"cannot concatenate results of type {type(left)} (per-chunk results "
        "must be tensors, ndarrays, dicts, lists or tuples)"
    )


def iter_with_free(
    iterable: Iterable[T],
    *,
    every: int = 25,
    collect: bool = True,
    label: str = "iteration",
) -> Iterable[T]:
    """Yield from ``iterable``, freeing memory every ``every`` items.

    Long eval loops otherwise rely purely on refcounting: CUDA caching
    allocator blocks and cyclic references (tensors captured by closures) can
    hold VRAM for the whole run.
    """
    for i, item in enumerate(iterable, start=1):
        yield item
        if every and i % every == 0:
            free_memory(collect=collect)
            logger.debug("%s: freed memory after %d items", label, i)
