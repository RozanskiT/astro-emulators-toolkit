from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, cast, runtime_checkable

import jax
import numpy as np

Batch = dict[str, Any]


@runtime_checkable
class DatasetProtocol(Protocol):
    def __len__(self) -> int: ...

    def get_batch(self, idx: np.ndarray) -> Batch: ...


@runtime_checkable
class DeviceBatchTransform(Protocol):
    def for_init(self, batch: Batch) -> Batch: ...

    def __call__(
        self,
        batch: Batch,
        *,
        train: bool,
        rng: jax.Array | None,
    ) -> Batch: ...


@dataclass(frozen=True)
class IdentityDeviceBatchTransform:
    def for_init(self, batch: Batch) -> Batch:
        return batch

    def __call__(
        self,
        batch: Batch,
        *,
        train: bool,
        rng: jax.Array | None,
    ) -> Batch:
        del train, rng
        return batch


@dataclass(frozen=True)
class FunctionalDeviceBatchTransform:
    fn: Callable[..., Batch]

    def for_init(self, batch: Batch) -> Batch:
        return self(batch, train=False, rng=None)

    def __call__(
        self,
        batch: Batch,
        *,
        train: bool,
        rng: jax.Array | None,
    ) -> Batch:
        return self.fn(batch, train=train, rng=rng)


DeviceBatchTransformLike = DeviceBatchTransform | Callable[..., Batch]


def call_device_batch_transform(
    device_batch_transform: DeviceBatchTransform | Callable[..., Batch],
    batch: Batch,
    *,
    train: bool,
    rng: jax.Array | None,
) -> Batch:
    return cast(Callable[..., Batch], device_batch_transform)(
        batch,
        train=train,
        rng=rng,
    )


def init_batch_via_device_transform(
    device_batch_transform: DeviceBatchTransform | Callable[..., Batch],
    batch: Batch,
) -> Batch:
    if hasattr(device_batch_transform, "for_init"):
        return cast(DeviceBatchTransform, device_batch_transform).for_init(batch)
    return call_device_batch_transform(
        device_batch_transform,
        batch,
        train=False,
        rng=None,
    )
