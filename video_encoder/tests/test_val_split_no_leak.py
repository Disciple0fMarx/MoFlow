"""Regression tests for the model-selection val-leak fix.

Historical Trainer fell back to the *test* loader when ``val_loader`` was not
passed, so the best checkpoint was selected on the test set and reported on
the same set — inflating final ADE/FDE. The fix carves a deterministic
validation split out of the TRAIN windows instead.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from trainer.denoising_model_trainers import build_val_loader_from_train


class _DummyDataset(Dataset):
    def __init__(self, n: int, offset: int = 0):
        self.items = [(i + offset) * 1.0 for i in range(n)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        return torch.tensor(self.items[i])


def _loader(ds: Dataset) -> DataLoader:
    return DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)


def test_val_split_holds_out_fraction_of_train() -> None:
    class Cfg:
        def get(self, k, d):
            return 0.25  # VAL_FRACTION

    ds = _DummyDataset(100)
    train_l = _loader(ds)
    val_l = build_val_loader_from_train(Cfg(), train_l, val_fraction=0.25, seed=0)
    assert len(val_l.dataset) == 25


def test_val_split_is_deterministic_given_seed() -> None:
    class Cfg:
        def get(self, k, d):
            return 0.2

    ds = _DummyDataset(100)
    a = build_val_loader_from_train(Cfg(), _loader(ds), val_fraction=0.2, seed=42)
    b = build_val_loader_from_train(Cfg(), _loader(ds), val_fraction=0.2, seed=42)
    assert list(a.dataset) == list(b.dataset)


def test_val_split_does_not_overlap_unambiguously() -> None:
    # Every item has a unique fingerprint; the held-out val items must be a
    # strict (deterministic) subset of the ORIGINAL train pool and contain no
    # duplicates.
    class Cfg:
        def get(self, k, d):
            return 0.3

    ds = _DummyDataset(50, offset=100)  # unique fingerprints 100..149
    val_l = build_val_loader_from_train(Cfg(), _loader(ds), val_fraction=0.3, seed=7)
    vals = [float(x) for x in val_l.dataset]
    assert len(vals) == len(set(vals)) == 15
    assert all(v in set(float(x) for x in ds) for v in vals)


def test_val_split_batch_pipeline_matches_train() -> None:
    class Cfg:
        def get(self, k, d):
            return 0.1

    ds = _DummyDataset(64)
    train_l = _loader(ds)
    val_l = build_val_loader_from_train(Cfg(), train_l, val_fraction=0.1, seed=0)
    # same collate_fn & batch_size, but not shuffled
    assert val_l.batch_size == train_l.batch_size
    assert val_l.collate_fn == train_l.collate_fn
    from torch.utils.data.sampler import SequentialSampler

    assert isinstance(val_l.sampler, SequentialSampler)