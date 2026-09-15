"""Python-side tests for the unfolding core (round-trip gates + input errors)."""

import math

import pytest

import nucleide.unfold as unfold

N_GROUPS = 6


def _synthetic_response(n_det: int, n_groups: int, width: float) -> list[list[float]]:
    """Deterministic synthetic response matrix (hand-built, never evaluated data)."""
    rows = []
    for i in range(n_det):
        center = i * (n_groups - 1) / max(n_det - 1, 1)
        rows.append([math.exp(-(((j - center) / width) ** 2)) + 1e-3 for j in range(n_groups)])
    return rows


def _log_midpoints(n_groups: int, lo: float, hi: float) -> list[float]:
    step = math.log(hi / lo) / n_groups
    return [math.exp(math.log(lo) + (j + 0.5) * step) for j in range(n_groups)]


def _maxwellian_plus_inv_e(midpoints: list[float], kt: float) -> list[float]:
    return [e * math.exp(-e / kt) + 1e-6 / e for e in midpoints]


def test_forward_fold_matches_hand_matvec() -> None:
    response = _synthetic_response(3, 4, 1.5)
    spectrum = [1.0, 2.0, 3.0, 4.0]
    rates = unfold.forward_fold(response, spectrum)
    for i, row in enumerate(response):
        assert rates[i] == pytest.approx(sum(r * p for r, p in zip(row, spectrum, strict=True)))


def test_exact_guess_is_a_fixed_point() -> None:
    response = _synthetic_response(4, 12, 2.0)
    midpoints = _log_midpoints(12, 1e-6, 10.0)
    truth = _maxwellian_plus_inv_e(midpoints, 2.53e-5)
    rates = unfold.forward_fold(response, truth)
    out = unfold.sandii(response, rates, truth)
    assert out["iterations"] == 1
    assert out["max_rel_change"] == 0.0
    assert out["spectrum"] == truth
    assert out["rates"] == pytest.approx(rates, rel=1e-12)
    assert out["rate_factors"] == pytest.approx([1.0] * len(rates), rel=1e-12)


def test_recovers_determined_system_from_biased_guess() -> None:
    response = _synthetic_response(N_GROUPS, N_GROUPS, 0.9)
    midpoints = _log_midpoints(N_GROUPS, 1e-6, 10.0)
    truth = _maxwellian_plus_inv_e(midpoints, 2.53e-5)
    rates = unfold.forward_fold(response, truth)
    guess = [3.0 * p for p in truth]
    out = unfold.sandii(response, rates, guess, tolerance=1e-12, max_iterations=10_000)
    assert out["spectrum"] == pytest.approx(truth, rel=1e-6)
    assert out["rate_factors"] == pytest.approx([1.0] * N_GROUPS, rel=1e-9)
    assert out["iterations"] >= 1


def test_underdetermined_round_trip_reproduces_rates() -> None:
    n_groups = 24
    response = _synthetic_response(6, n_groups, 3.0)
    midpoints = _log_midpoints(n_groups, 1e-6, 12.0)
    truth = _maxwellian_plus_inv_e(midpoints, 2.53e-5)
    rates = unfold.forward_fold(response, truth)
    # Non-uniform bias: part lies in the response null space and must keep
    # the guess; the recoverable part must still converge.
    guess = [p * (1.3 + 0.7 * (j % 5) / 4.0) for j, p in enumerate(truth)]

    def max_rel_err(a: list[float], b: list[float]) -> float:
        return max(abs(x - y) / y for x, y in zip(a, b, strict=True))

    out = unfold.sandii(response, rates, guess, tolerance=1e-9, max_iterations=50_000)
    assert out["rate_factors"] == pytest.approx([1.0] * 6, rel=1e-6)
    assert max_rel_err(out["spectrum"], truth) < max_rel_err(guess, truth) / 2.0
    assert all(p > 0.0 for p in out["spectrum"])


def test_irdff_ii_analytical_benchmark_shapes_recover() -> None:
    # IRDFF-II (Trkov et al., Nucl. Data Sheets 163 (2020) 1, arXiv:1909.03336)
    # names analytical benchmark-field shapes; three are reused here as
    # published facts with citation. The library's tabulated group spectra
    # are IAEA-copyright data files and are deliberately NOT used.
    n_groups = 6
    midpoints = _log_midpoints(n_groups, 1e-9, 20.0)
    thermal_kt = 8.617333262e-5 * 293.6  # 293.6 K, eV -> MeV
    shapes = {
        "thermal": [e * math.exp(-e / thermal_kt) + 1e-30 for e in midpoints],
        "1/E": [1.0 / e + 1e-30 for e in midpoints],
        "25keV": [math.sqrt(e) * math.exp(-e / 2.5e-2) + 1e-30 for e in midpoints],
    }
    for shape in shapes.values():
        response = _synthetic_response(n_groups, n_groups, 0.8)
        rates = unfold.forward_fold(response, shape)
        guess = [5.0 * p for p in shape]
        out = unfold.sandii(response, rates, guess, tolerance=1e-11, max_iterations=100_000)
        assert out["spectrum"] == pytest.approx(shape, rel=1e-5)


def test_non_convergence_is_a_hard_fail() -> None:
    response = _synthetic_response(4, 8, 1.5)
    midpoints = _log_midpoints(8, 1e-6, 10.0)
    truth = _maxwellian_plus_inv_e(midpoints, 2.53e-5)
    rates = unfold.forward_fold(response, truth)
    guess = [2.0 * p for p in truth]
    with pytest.raises(ValueError, match="did not converge"):
        unfold.sandii(response, rates, guess, tolerance=1e-12, max_iterations=1)


def test_validation_errors_name_the_cause() -> None:
    response = _synthetic_response(2, 3, 1.0)
    rates = [1.0, 2.0]
    guess = [1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="shape mismatch"):
        unfold.sandii(response, [1.0], guess)
    with pytest.raises(ValueError, match="non-finite"):
        unfold.sandii(response, [1.0, float("nan")], guess)
    with pytest.raises(ValueError, match="negative entry"):
        unfold.sandii([[1.0, -1.0, 1.0], [1.0, 1.0, 1.0]], rates, guess)
    with pytest.raises(ValueError, match="strictly positive"):
        unfold.sandii(response, rates, [1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="tolerance"):
        unfold.sandii(response, rates, guess, tolerance=0.0)
    with pytest.raises(ValueError, match="max_iterations"):
        unfold.sandii(response, rates, guess, max_iterations=0)
    with pytest.raises(ValueError, match="unreachable"):
        unfold.sandii([[0.0, 0.0, 0.0]], [1.0], guess)
    with pytest.raises(ValueError, match="shape mismatch"):
        unfold.forward_fold(response, [1.0])
