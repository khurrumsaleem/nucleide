"""Variance-reduction tools (backed by the `nucleide-vr-tools` crate)."""

from nucleide._internal import (
    AliasTable,
    KdeSampler,
    MagicOutput,
    MeshSourceSampler,
    emit_openmc_weight_windows,
    emit_serpent_wwin,
    magic,
    magic_with,
)

__all__ = [
    "magic",
    "magic_with",
    "MagicOutput",
    "emit_openmc_weight_windows",
    "emit_serpent_wwin",
    "AliasTable",
    "MeshSourceSampler",
    "KdeSampler",
]
