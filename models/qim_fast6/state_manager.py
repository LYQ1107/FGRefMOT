"""Lifecycle-safe state helpers for the fast-six QIMs.

The tracker stores these tensors as ordinary ``Instances`` fields.  They are
therefore indexed, concatenated, and moved between devices with the same
track lifecycle as the original fields.  No helper in this file accepts GT
IDs, labels, or referring targets as an input.
"""

from __future__ import annotations

import torch


def _device_dtype(instances, dtype=None):
    reference = instances.query_pos
    return reference.device, dtype or reference.dtype


def ensure_tensor_field(instances, name, shape, dtype=None):
    """Return an existing field or create a zero field with per-track shape."""

    if instances.has(name):
        value = instances.get(name)
        if tuple(value.shape) != tuple(shape):
            raise ValueError(
                "{} has shape {}, expected {}".format(name, tuple(value.shape), tuple(shape))
            )
        return value
    device, default_dtype = _device_dtype(instances, dtype)
    value = torch.zeros(shape, device=device, dtype=default_dtype)
    setattr(instances, name, value)
    return value


def ensure_bool_field(instances, name):
    if instances.has(name):
        return instances.get(name).bool()
    device, _ = _device_dtype(instances)
    value = torch.zeros((len(instances),), device=device, dtype=torch.bool)
    setattr(instances, name, value)
    return value


def ensure_long_field(instances, name):
    if instances.has(name):
        return instances.get(name).long()
    device, _ = _device_dtype(instances)
    value = torch.zeros((len(instances),), device=device, dtype=torch.long)
    setattr(instances, name, value)
    return value


def ensure_box_field(instances, name):
    return ensure_tensor_field(
        instances,
        name,
        (len(instances), 4),
        dtype=torch.float32,
    )
