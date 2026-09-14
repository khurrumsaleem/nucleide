"""Round-trip tests for the legacy SDEF reader (synthetic cards only)."""

import pytest

import nucleide.mcnp as mcnp
import nucleide.spectroscopy as sp


def _emit_single() -> str:
    return sp.sdef_decay_source([(0.662, 2.0)])[1]


def _emit_multi() -> str:
    return sp.sdef_decay_source([(1.17, 1.0), (0.662, 2.0), (1.33, 1.0)])[1]


def _emit_beam() -> str:
    return sp.sdef_decay_source([(0.662, 2.0)], z=3.0, w=1.0, weight=0.5, particle="Photon")[1]


def _emit_wrapped() -> str:
    lines = [(0.123456 * i, float(i)) for i in range(1, 13)]
    return sp.sdef_decay_source(lines)[1]


class TestSdefRoundTrip:
    def test_single_line_isotropic_byte_identical(self) -> None:
        card = _emit_single()
        parsed = mcnp.parse_sdef(card)
        assert parsed["card"] == card
        assert parsed["pos"] == "0 0 0"
        assert parsed["erg"] == "0.662"
        assert parsed["wgt"] == "1"
        assert parsed["par"] == "n"
        assert parsed["distributions"] == []
        assert parsed["ignored"] == []

    def test_multi_line_distribution_byte_identical(self) -> None:
        card = _emit_multi()
        parsed = mcnp.parse_sdef(card)
        assert parsed["card"] == card
        assert parsed["erg"] == "D1"
        (dist,) = parsed["distributions"]
        assert dist["number"] == "1"
        assert dist["si_option"] == "L"
        assert dist["si"] == "0.662 1.17 1.33"
        assert dist["sp_option"] == "D"
        assert dist["sp"] == "0.5 0.25 0.25"
        assert dist["sb"] == ""

    def test_beam_joint_vec_dir_line_byte_identical(self) -> None:
        card = _emit_beam()
        parsed = mcnp.parse_sdef(card)
        assert parsed["card"] == card
        assert parsed["vec"] == "0 0 1"
        assert parsed["dir"] == "1"
        assert parsed["par"] == "p"

    def test_wrapped_distribution_lists_byte_identical(self) -> None:
        card = _emit_wrapped()
        assert all(len(line) <= 80 for line in card.splitlines())
        assert any(
            line.startswith("      ") and line.strip()[:1].isdigit() for line in card.splitlines()
        ), "emitter wraps SI1/SP1 onto aligned continuations"
        parsed = mcnp.parse_sdef(card)
        assert parsed["card"] == card
        assert len(parsed["distributions"]) == 1
        dist = parsed["distributions"][0]
        assert len(dist["si"].split()) == 12
        assert len(dist["sp"].split()) == 12

    def test_lowercase_input_normalizes_to_canonical_emit(self) -> None:
        parsed = mcnp.parse_sdef("sdef pos=0 0 0\n     erg=d1\nsi1 l 0.662\nsp1 d 1")
        assert parsed["card"] == "SDEF POS=0 0 0\n     ERG=D1\nSI1 L 0.662\nSP1 D 1"

    def test_full_card_fields_not_in_emitter_dialect(self) -> None:
        # Already in canonical emission order (POS would sit on the SDEF head
        # line; every other keyword gets its own five-space continuation).
        card = (
            "SDEF\n"
            "     CELL=4\n"
            "     SURF=2\n"
            "     NRM=1\n"
            "     TME=D2\n"
            "SI2 L 0 1\n"
            "SP2 D 0.5 0.5\n"
            "SB2 D 1 1"
        )
        parsed = mcnp.parse_sdef(card)
        assert parsed["card"] == card
        assert parsed["cell"] == "4"
        assert parsed["surf"] == "2"
        assert parsed["nrm"] == "1"
        assert parsed["tme"] == "D2"
        assert parsed["pos"] == ""
        (dist,) = parsed["distributions"]
        assert dist["si"] == "0 1"
        assert dist["sp"] == "0.5 0.5"
        assert dist["sb"] == "1 1"


class TestSdefDeckWiring:
    def test_deck_sdef_view_and_absent_sdef(self) -> None:
        deck = mcnp.parse_deck("msg\ntitle\n1 0 -1\n\n1 so 1.0\n\n" + _emit_multi() + "\n")
        view = deck.sdef
        assert view is not None
        assert view["card"] == _emit_multi()
        assert view["erg"] == "D1"

        bare = mcnp.parse_deck("msg\ntitle\n1 0 -1\n\n1 so 1.0\n\nmode n\n")
        assert bare.sdef is None

    def test_deck_validation_rejects_dangling_distribution(self) -> None:
        deck = mcnp.parse_deck("msg\ntitle\n1 0 -1\n\n1 so 1.0\n\nSDEF ERG=D2\n")
        with pytest.raises(ValueError, match="SI2"):
            deck.validate()


class TestSdefErrors:
    def test_missing_sdef_card(self) -> None:
        with pytest.raises(ValueError, match="no SDEF card"):
            mcnp.parse_sdef("SI1 L 0.662")

    def test_non_discrete_form_rejected(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            mcnp.parse_sdef("SDEF ERG=D1\nSI1 H 0.662 1.17")

    def test_dangling_reference_rejected(self) -> None:
        with pytest.raises(ValueError, match="SI2"):
            mcnp.parse_sdef("SDEF ERG=D2\nSI1 L 0.662")

    def test_orphan_sp_rejected(self) -> None:
        with pytest.raises(ValueError, match="SP1"):
            mcnp.parse_sdef("SDEF POS=0 0 0\nSP1 D 0.5")

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="entries"):
            mcnp.parse_sdef("SDEF ERG=D1\nSI1 L 1.0 2.0\nSP1 D 1.0")

    def test_unknown_keywords_are_drift_notes(self) -> None:
        parsed = mcnp.parse_sdef("SDEF POS=0 0 0 ERG=0.662 AXS=0 0 1")
        assert parsed["ignored"] == ["AXS=0 0 0 1"]
        assert parsed["erg"] == "0.662"
