"""Neutron spectrum unfolding (backed by the `nucleide-unfold` crate)."""

from typing import Any

from nucleide._internal import unfold_forward_fold, unfold_sandii

__all__ = ["sandii", "forward_fold"]


def sandii(
    response: list[list[float]],
    rates: list[float],
    guess: list[float],
    *,
    tolerance: float = 1e-3,
    max_iterations: int = 200,
) -> dict[str, Any]:
    """SAND-II iterative spectral adjustment (McElroy et al., AFWL-TR-67-41, 1967).

    ``response`` holds one row per detector/reaction (all rows one value per
    energy group), ``rates`` the measured rate per detector, and ``guess``
    one strictly positive value per energy group. The guess is adjusted
    until folding it through the response matrix reproduces the rates.
    ``tolerance`` is the largest per-group relative change between
    successive adjustments the run converges under; ``max_iterations`` is
    the explicit adjustment cap — exhausting it raises ``ValueError``
    (non-convergence is a hard fail, never a silent partial spectrum).
    Returns ``spectrum``/``rates``/``rate_factors``/``iterations``/
    ``tolerance``/``max_rel_change``.
    """
    return unfold_sandii(response, rates, guess, tolerance, max_iterations)


def forward_fold(response: list[list[float]], spectrum: list[float]) -> list[float]:
    """Fold a spectrum through a response matrix (one rate per detector row)."""
    return unfold_forward_fold(response, spectrum)
