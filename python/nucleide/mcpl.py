"""MCPL particle-list interchange (backed by the `nucleide-mcpl-io` crate).

Thin facade over the flat `_internal` extension: use :func:`read_mcpl` to
parse a file (`.gz` reads through gzip transparently) and :func:`write_mcpl`
to emit one from a header dict plus particle dicts. Particle units follow the
published format (kinetic energy in MeV, position in cm, time in ms).
SSW↔MCPL conversion (SSW-PDG table: neutron/gamma/electron/positron/proton)
runs through :func:`ssw2mcpl` and :func:`mcpl2ssw`: SSW tracks need one
explicit surface id + particle kind per track, and `mcpl2ssw` clones a
reference SSW header with `nrss`/`np1`/`orignp1` patched (`niss` passes
through unless `niss=` stamps an override). PDG codes outside the table are
errors, never silent skips. Opt-ins: `ssw2mcpl` `polarisation` /
`universal_pdg` / `universal_weight`; `mcpl2ssw` `force_cs_to_one` / `niss` /
`allow_polarisation`.

Particle-list utilities: :func:`merge_mcpl` concatenates compatible files
(first-file header wins plus a provenance comment; `stat:sum` sums are never
synthesized; mixed precision promotes to double), :func:`extract_mcpl`
writes an index-range or predicate subset with the source header preserved
verbatim, :func:`mcpl_stats` returns record counts, energy moments, and a
PDG histogram, and :func:`repair_mcpl` recomputes the particle count of a
file that was never properly closed.
Cross-tool byte compatibility beyond self-consistent round-trips is
oracle-gated (see `validation/mcpl_vs_refs.py`).
"""

from nucleide._internal import (
    McplFile,
    extract_mcpl,
    mcpl2ssw,
    mcpl_stats,
    mcpl_statsum_comment,
    mcpl_statsum_validate,
    merge_mcpl,
    read_mcpl,
    repair_mcpl,
    ssw2mcpl,
    write_mcpl,
)

__all__ = [
    "McplFile",
    "read_mcpl",
    "write_mcpl",
    "merge_mcpl",
    "extract_mcpl",
    "mcpl_stats",
    "repair_mcpl",
    "ssw2mcpl",
    "mcpl2ssw",
    "mcpl_statsum_validate",
    "mcpl_statsum_comment",
]
