"""CSG translation tests: MCNP decks to OpenMC/Serpent/PHITS/GDML (synthetic fixtures)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import nucleide

FIXTURES = Path(__file__).parent.parent / "fixtures" / "mcnp" / "inp"

GO_FIXTURES = [
    "deck_csg_sphere_box.txt",
    "deck_csg_rpp.txt",
    "deck_csg_rcc.txt",
    "deck_csg_cylinders.txt",
    "deck_csg_complement.txt",
    "deck_csg_universe_fill.txt",
    "deck_csg_universe_data.txt",
    "deck_csg_lattice_rect.txt",
]


def _translate(name: str) -> tuple[str, list[dict[str, str]]]:
    return nucleide.mcnp.read_csg_to_openmc(str(FIXTURES / name))


def _regions(xml: str) -> dict[str, str]:
    root = ET.fromstring(xml)
    assert root.tag == "geometry"
    return {c.get("id", ""): c.get("region", "") for c in root.findall("cell")}


def _surfaces(xml: str) -> dict[str, dict[str, str]]:
    root = ET.fromstring(xml)
    return {
        s.get("id", ""): {"type": s.get("type", ""), "coeffs": s.get("coeffs", "")}
        for s in root.findall("surface")
    }


class TestCsgFixtures:
    def test_go_fixtures_translate(self) -> None:
        for name in GO_FIXTURES:
            xml, drift = _translate(name)
            root = ET.fromstring(xml)
            assert root.tag == "geometry"
            assert root.findall("surface")
            assert root.findall("cell")
            assert isinstance(drift, list)

    def test_sphere_box_regions(self) -> None:
        xml, _ = _translate("deck_csg_sphere_box.txt")
        assert _regions(xml) == {
            "1": "-1",
            "2": "1 2 -3 4 -5 6 -7",
            "3": "-2 | 3 | -4 | 5 | -6 | 7",
        }
        surfs = _surfaces(xml)
        assert surfs["1"] == {"type": "sphere", "coeffs": "0 0 0 5"}
        assert surfs["2"] == {"type": "x-plane", "coeffs": "-10"}

    def test_rpp_expands_to_six_planes(self) -> None:
        xml, drift = _translate("deck_csg_rpp.txt")
        surfs = _surfaces(xml)
        assert len(surfs) == 6
        assert [s["type"] for s in surfs.values()].count("x-plane") == 2
        assert _regions(xml)["1"] == "2 -3 4 -5 6 -7"
        actions = [d["action"] for d in drift]
        assert "macrobody-expansion" in actions

    def test_rcc_expands_to_cylinder_and_caps(self) -> None:
        xml, drift = _translate("deck_csg_rcc.txt")
        surfs = _surfaces(xml)
        assert surfs["2"] == {"type": "z-cylinder", "coeffs": "0 0 2"}
        assert surfs["3"] == {"type": "z-plane", "coeffs": "-5"}
        assert surfs["4"] == {"type": "z-plane", "coeffs": "5"}
        assert _regions(xml)["1"] == "-2 3 -4"
        assert any(d["action"] == "macrobody-expansion" and d["scope"] == "surface" for d in drift)

    def test_complement_inlines(self) -> None:
        xml, drift = _translate("deck_csg_complement.txt")
        assert _regions(xml)["2"] == "1 -2"
        assert any(d["action"] == "complement-expansion" for d in drift)

    def test_universe_fill_regions(self) -> None:
        for name in ("deck_csg_universe_fill.txt", "deck_csg_universe_data.txt"):
            xml, drift = _translate(name)
            root = ET.fromstring(xml)
            by_id = {c.get("id"): c for c in root.findall("cell")}
            assert by_id["1"].get("universe") == "1"
            assert by_id["1"].get("material") == "1"
            assert by_id["2"].get("universe") == "0"
            assert by_id["2"].get("fill") == "1"
            assert by_id["2"].get("material") is None
            actions = [d["action"] for d in drift]
            assert "universe-assigned" in actions
            assert "fill-applied" in actions

    def test_rect_lattice_emitted(self) -> None:
        xml, drift = _translate("deck_csg_lattice_rect.txt")
        root = ET.fromstring(xml)
        lat = root.find("lattice")
        assert lat is not None
        assert lat.get("type") == "rectangular"
        cells = root.findall("cell")
        assert lat.get("id") not in {c.get("id") for c in cells}
        assert lat.findtext("dimension") == "2 2 2"
        assert lat.findtext("lower_left") == "0 0 0"
        assert lat.findtext("pitch") == "2 2 2"
        # MCNP `k, j, i` (i fastest) maps to z ascending, y descending, x ascending.
        universes = lat.findtext("universes")
        assert universes == "3 4 1 2 7 8 5 6"
        defined = {c.get("universe", "0") for c in cells}
        assert set((universes or "").split()) <= defined
        lattice_cell = {c.get("id"): c for c in cells}["10"]
        assert lattice_cell.get("fill") == lat.get("id")
        assert lattice_cell.get("material") is None
        assert any(d["action"] == "lattice-emitted" and d["target"] == "10" for d in drift)

    def test_region_ids_reference_defined_surfaces(self) -> None:
        for name in GO_FIXTURES:
            xml, _ = _translate(name)
            root = ET.fromstring(xml)
            known = {s.get("id") for s in root.findall("surface")}
            for cell in root.findall("cell"):
                region = cell.get("region", "")
                ids = {
                    tok.lstrip("~+-")
                    for tok in region.replace("|", " ").replace("(", " ").replace(")", " ").split()
                }
                assert ids <= known, (name, region)

    def test_material_stub(self) -> None:
        xml, _ = _translate("deck_csg_sphere_box.txt")
        root = ET.fromstring(xml)
        mats = {c.get("id"): c.get("material") for c in root.findall("cell")}
        assert mats == {"1": "1", "2": "void", "3": "void"}

    def test_parse_text_matches_file(self) -> None:
        path = FIXTURES / "deck_csg_rpp.txt"
        xml_file, _ = nucleide.mcnp.read_csg_to_openmc(str(path))
        xml_text, _ = nucleide.mcnp.parse_csg_to_openmc(path.read_text())
        assert xml_text == xml_file


class TestCsgLoudErrors:
    def test_union_complement_rejected(self) -> None:
        with pytest.raises(ValueError, match="too complex"):
            nucleide.mcnp.read_csg_to_openmc(str(FIXTURES / "deck_csg_complement_reject.txt"))

    def test_cone_rejected(self) -> None:
        deck = "msg\ntitle\n1 1 -1.0 -1\n\n1 kz 0 0 0 1 1\n\nm1 1001 1.0\n"
        with pytest.raises(ValueError, match="no v1 mapping"):
            nucleide.mcnp.parse_csg_to_openmc(deck)

    def test_canted_rcc_rejected(self) -> None:
        deck = "msg\ntitle\n1 1 -1.0 -1\n\n1 rcc 0 0 0 1 1 1 2\n\nm1 1001 1.0\n"
        with pytest.raises(ValueError, match="out of v1 scope"):
            nucleide.mcnp.parse_csg_to_openmc(deck)

    def test_lattice_rejected(self) -> None:
        deck = "msg\ntitle\n1 0 -1 lat=1 fill=0\n\n1 so 10\n\n"
        with pytest.raises(ValueError, match="out of v1 scope"):
            nucleide.mcnp.parse_csg_to_openmc(deck)

    def test_lattice_loud_cases_all_directions(self) -> None:
        """Hex LAT=2, 0-holes, and non-RPP-bounded lattice cells stay loud."""
        cells = (
            "1 1 -1.0 -1 u=1\n2 1 -1.0 -2 u=2\n3 1 -1.0 -3 u=3\n4 1 -1.0 -4 u=4\n"
            "10 0 -100 {params}\n20 0 #10"
        )
        surfs = "1 sph 1 1 1 0.5\n2 sph 3 1 1 0.5\n3 sph 1 3 1 0.5\n4 sph 3 3 1 0.5\n{bounds}"
        rpp = "100 rpp 0 4 0 4 0 2"
        cases = [
            ("lat=2 fill=0:1 0:1 0:0 1 2 3 4", rpp, "lat=2"),
            ("lat=1 fill=0:1 0:1 0:0 1 0 3 4", rpp, "0-hole"),
            ("lat=1 fill=0:1 0:1 0:0 1 2 3 4", "100 so 10", "lattice bounds"),
        ]
        for params, bounds, match in cases:
            deck = (
                "msg\ntitle\n"
                + cells.format(params=params)
                + "\n\n"
                + surfs.format(bounds=bounds)
                + "\n\nm1 92235 1.0\n"
            )
            for parse in (
                nucleide.mcnp.parse_csg_to_openmc,
                nucleide.mcnp.parse_csg_to_serpent,
                nucleide.mcnp.parse_csg_to_phits,
                nucleide.mcnp.parse_csg_to_gdml,
            ):
                with pytest.raises(ValueError, match=match):
                    parse(deck)

    def test_universe_notrunc_rejected(self) -> None:
        deck = "msg\ntitle\n1 0 -1 u=-1\n\n1 so 10\n\n"
        with pytest.raises(ValueError, match="out of v1 scope"):
            nucleide.mcnp.parse_csg_to_openmc(deck)

    def test_plain_universe_translates(self) -> None:
        xml, _ = nucleide.mcnp.parse_csg_to_openmc("msg\ntitle\n1 0 -1 u=0\n\n1 so 10\n\n")
        found = ET.fromstring(xml).find("cell")
        assert found is not None
        assert found.get("universe") == "0"

    def test_tally_rejected(self) -> None:
        deck = "msg\ntitle\n1 0 -1\n\n1 so 10\n\nf4:n 1\n"
        with pytest.raises(ValueError, match="out of v1 scope"):
            nucleide.mcnp.parse_csg_to_openmc(deck)

    def test_source_rejected(self) -> None:
        deck = "msg\ntitle\n1 0 -1\n\n1 so 10\n\nsdef pos=0 0 0\n"
        with pytest.raises(ValueError, match="out of v1 scope"):
            nucleide.mcnp.parse_csg_to_openmc(deck)

    def test_trcl_rejected(self) -> None:
        deck = "msg\ntitle\n1 0 -1 trcl=1\n\n1 so 10\n\ntr1 0 0 0\n"
        with pytest.raises(ValueError, match="out of v1 scope"):
            nucleide.mcnp.parse_csg_to_openmc(deck)

    def test_reflective_periodic_conflict_rejected(self) -> None:
        deck = "msg\ntitle\n1 0 -1\n\n*1 -2 pz 0\n2 pz 5\n\n"
        with pytest.raises(ValueError, match="both reflective and periodic"):
            nucleide.mcnp.parse_csg_to_openmc(deck)

    def test_periodic_divergent_rejected(self) -> None:
        deck = "msg\ntitle\n1 0 -1\n\n1 -2 pz 0\n2 -3 pz 5\n3 pz 9\n\n"
        with pytest.raises(ValueError, match="ambiguous"):
            nucleide.mcnp.parse_csg_to_openmc(deck)

    def test_unknown_surface_rejected(self) -> None:
        deck = "msg\ntitle\n1 0 -99\n\n1 so 10\n\n"
        with pytest.raises(ValueError, match="missing surface"):
            nucleide.mcnp.parse_csg_to_openmc(deck)

    def test_unknown_cell_rejected(self) -> None:
        deck = "msg\ntitle\n1 0 -1 #-99\n\n1 so 10\n\n"
        with pytest.raises(ValueError, match="missing cell"):
            nucleide.mcnp.parse_csg_to_openmc(deck)

    def test_reflecting_and_periodic(self) -> None:
        xml, drift = nucleide.mcnp.parse_csg_to_openmc(
            "msg\ntitle\n1 0 -1 2 -3\n\n*1 so 10\n2 -3 pz 0\n3 pz 5\n\n"
        )
        root = ET.fromstring(xml)
        by_id = {s.get("id"): s for s in root.findall("surface")}
        assert by_id["1"].get("boundary") == "reflective"
        assert by_id["2"].get("boundary") == "periodic"
        assert by_id["2"].get("periodic_surface_id") == "3"
        assert by_id["3"].get("periodic_surface_id") == "2"
        assert any(d["action"] == "periodic-link" for d in drift)


class TestCsgSerpent:
    def test_go_fixtures_emit_cards(self) -> None:
        for name in GO_FIXTURES:
            text, drift = nucleide.mcnp.read_csg_to_serpent(str(FIXTURES / name))
            assert text.startswith("% Serpent geometry")
            assert "surf " in text and "cell " in text
            assert isinstance(drift, list)

    def test_sphere_box_cards(self) -> None:
        text, _ = nucleide.mcnp.read_csg_to_serpent(str(FIXTURES / "deck_csg_sphere_box.txt"))
        lines = [ln for ln in text.splitlines() if not ln.startswith("%")]
        assert "surf 1 sph 0 0 0 5" in lines
        assert "surf 2 px -10" in lines
        assert "cell 1 0 m1 -1" in lines
        assert "cell 2 0 void 1 2 -3 4 -5 6 -7" in lines
        assert "cell 3 0 void -2 : 3 : -4 : 5 : -6 : 7" in lines

    def test_rpp_is_native_cuboid(self) -> None:
        text, drift = nucleide.mcnp.read_csg_to_serpent(str(FIXTURES / "deck_csg_rpp.txt"))
        assert "cuboid" in text
        assert not any(d["action"] == "macrobody-expansion" for d in drift)

    def test_universe_fill_cards(self) -> None:
        for name in ("deck_csg_universe_fill.txt", "deck_csg_universe_data.txt"):
            text, drift = nucleide.mcnp.read_csg_to_serpent(str(FIXTURES / name))
            assert "cell 1 1 m1 -1" in text.splitlines()
            assert "cell 2 0 fill 1 -2" in text.splitlines()
            actions = [d["action"] for d in drift]
            assert "fill-applied" in actions

    def test_rect_lattice_lat_card(self) -> None:
        text, drift = nucleide.mcnp.read_csg_to_serpent(str(FIXTURES / "deck_csg_lattice_rect.txt"))
        lines = text.splitlines()
        # Cuboidal type 11: centre, counts, pitches, then the same
        # z-ascending / y-descending / x-ascending universe order as OpenMC.
        assert "lat 21 11 2 2 2 2 2 2 2 2 2 3 4 1 2 7 8 5 6" in lines
        assert "cell 10 0 fill 21 -100" in lines
        assert any(d["action"] == "lattice-emitted" and d["target"] == "10" for d in drift)

    def test_complement_passes_through(self) -> None:
        text, _ = nucleide.mcnp.read_csg_to_serpent(str(FIXTURES / "deck_csg_complement.txt"))
        assert "cell 2 0 void #1 -2" in text.splitlines()

    def test_parse_text_matches_file(self) -> None:
        path = FIXTURES / "deck_csg_sphere_box.txt"
        file_text, _ = nucleide.mcnp.read_csg_to_serpent(str(path))
        text, _ = nucleide.mcnp.parse_csg_to_serpent(path.read_text())
        assert text == file_text

    def test_boundaries_rejected(self) -> None:
        with pytest.raises(ValueError, match="no verified serpent mapping"):
            nucleide.mcnp.parse_csg_to_serpent("msg\ntitle\n1 0 -1\n\n*1 so 10\n\n")
        with pytest.raises(ValueError, match="no verified serpent mapping"):
            nucleide.mcnp.parse_csg_to_serpent("msg\ntitle\n1 0 *-1\n\n1 so 10\n\n")
        with pytest.raises(ValueError, match="no verified serpent mapping"):
            nucleide.mcnp.parse_csg_to_serpent("msg\ntitle\n1 0 -1 2\n\n1 -2 pz 0\n2 pz 5\n\n")

    def test_out_of_scope_shared(self) -> None:
        with pytest.raises(ValueError, match="no v1 mapping"):
            nucleide.mcnp.parse_csg_to_serpent(
                "msg\ntitle\n1 1 -1.0 -1\n\n1 kz 0 0 0 1 1\n\nm1 1001 1.0\n"
            )
        with pytest.raises(ValueError, match="out of v1 scope"):
            nucleide.mcnp.parse_csg_to_serpent("msg\ntitle\n1 0 -1 u=-1\n\n1 so 10\n\n")


class TestCsgPhits:
    def test_go_fixtures_emit_sections(self) -> None:
        for name in GO_FIXTURES:
            text, drift = nucleide.mcnp.read_csg_to_phits(str(FIXTURES / name))
            assert "[ Surface ]" in text and "[ Cell ]" in text
            assert isinstance(drift, list)

    def test_sphere_box_sections(self) -> None:
        text, _ = nucleide.mcnp.read_csg_to_phits(str(FIXTURES / "deck_csg_sphere_box.txt"))
        lines = text.splitlines()
        assert "1  SO  5" in lines
        assert "2  PX  -10" in lines
        assert "1  1  -10  -1" in lines
        assert "2  0  1 2 -3 4 -5 6 -7" in lines

    def test_universe_fill_params(self) -> None:
        for name in ("deck_csg_universe_fill.txt", "deck_csg_universe_data.txt"):
            text, drift = nucleide.mcnp.read_csg_to_phits(str(FIXTURES / name))
            assert "1  1  -10  -1  U=1" in text.splitlines()
            assert "2  0  -2  FILL=1" in text.splitlines()
            assert any(d["action"] == "fill-applied" for d in drift)

    def test_rect_lattice_keeps_matrix_fill(self) -> None:
        text, drift = nucleide.mcnp.read_csg_to_phits(str(FIXTURES / "deck_csg_lattice_rect.txt"))
        # PHITS keeps LAT=1 with ranges plus the universe list in MCNP
        # order verbatim (x fastest).
        assert "10  0  -100  LAT=1  FILL=0:1 0:1 0:1 1 2 3 4 5 6 7 8" in text.splitlines()
        assert any(d["action"] == "lattice-emitted" and d["target"] == "10" for d in drift)

    def test_outer_void_and_reflective(self) -> None:
        text, drift = nucleide.mcnp.read_csg_to_phits(str(FIXTURES / "deck_csg_complement.txt"))
        assert "2  -1  #1 -2" in text.splitlines()
        assert any(d["action"] == "outer-void-assigned" for d in drift)
        text, _ = nucleide.mcnp.parse_csg_to_phits("msg\ntitle\n1 0 -1\n\n*1 so 10\n\n")
        assert "*1  SO  10" in text.splitlines()

    def test_parse_text_matches_file(self) -> None:
        path = FIXTURES / "deck_csg_sphere_box.txt"
        file_text, _ = nucleide.mcnp.read_csg_to_phits(str(path))
        text, _ = nucleide.mcnp.parse_csg_to_phits(path.read_text())
        assert text == file_text

    def test_periodic_rejected(self) -> None:
        with pytest.raises(ValueError, match="no phits spelling"):
            nucleide.mcnp.parse_csg_to_phits("msg\ntitle\n1 0 -1 2\n\n1 -2 pz 0\n2 pz 5\n\n")
        with pytest.raises(ValueError, match="no v1 mapping"):
            nucleide.mcnp.parse_csg_to_phits(
                "msg\ntitle\n1 1 -1.0 -1\n\n1 kz 0 0 0 1 1\n\nm1 1001 1.0\n"
            )


class TestCsgGdml:
    def test_go_fixtures_emit_documents(self) -> None:
        for name in GO_FIXTURES:
            xml, drift = nucleide.mcnp.read_csg_to_gdml(str(FIXTURES / name))
            root = ET.fromstring(xml)
            assert root.tag == "gdml"
            assert root.get("version") == "3.1.7"
            for section in ("define", "materials", "solids", "structure", "setup"):
                assert root.find(section) is not None, (name, section)
            assert isinstance(drift, list)

    def test_sphere_box_solids(self) -> None:
        xml, drift = nucleide.mcnp.read_csg_to_gdml(str(FIXTURES / "deck_csg_sphere_box.txt"))
        root = ET.fromstring(xml)
        solids_sec = root.find("solids")
        assert solids_sec is not None
        solids = {s.get("name"): s for s in solids_sec}
        # Sphere: origin-centered orb; planes: one cube per surface+sense.
        # GDML lengths are millimetres: 5 cm -> "50".
        assert solids["sl1"].tag == "orb"
        assert solids["sl1"].get("r") == "50"
        assert solids["sl2p"].tag == "box"
        assert solids["sl2m"].tag == "box"
        # Cell 2 = box minus sphere (its region lists the positive sphere).
        names = [e.tag for e in solids_sec]
        assert "intersection" in names and "subtraction" in names
        actions = [d["action"] for d in drift]
        assert "halfspace-bounded" in actions
        assert "material-stub" in actions
        structure = root.find("structure")
        assert structure is not None
        volumes = {v.get("name"): v for v in structure.findall("volume")}
        vol1_solidref = volumes["vol1"].find("solidref")
        world_solidref = volumes["world"].find("solidref")
        assert vol1_solidref is not None and vol1_solidref.get("ref") == "sl1"
        assert world_solidref is not None and world_solidref.get("ref") == "bigbox"

    def test_rpp_is_native_box_without_expansion(self) -> None:
        xml, drift = nucleide.mcnp.read_csg_to_gdml(str(FIXTURES / "deck_csg_rpp.txt"))
        root = ET.fromstring(xml)
        solids_sec = root.find("solids")
        assert solids_sec is not None
        solids = {s.get("name"): s for s in solids_sec}
        assert solids["sl1"].tag == "box"
        assert solids["sl1"].get("x") == "100"
        assert not any(d["action"] == "macrobody-expansion" for d in drift)

    def test_rcc_is_finite_tube_with_caps(self) -> None:
        xml, drift = nucleide.mcnp.read_csg_to_gdml(str(FIXTURES / "deck_csg_rcc.txt"))
        root = ET.fromstring(xml)
        solids_sec = root.find("solids")
        assert solids_sec is not None
        solids = {s.get("name"): s for s in solids_sec}
        tube = solids["sl1t"]
        assert tube.tag == "tube"
        assert tube.get("rmax") == "20"
        assert tube.get("z") == "100"
        assert any(d["action"] == "macrobody-expansion" for d in drift)

    def test_cylinders_rotate_x_and_y_tubes(self) -> None:
        xml, _ = nucleide.mcnp.read_csg_to_gdml(str(FIXTURES / "deck_csg_cylinders.txt"))
        root = ET.fromstring(xml)
        solids = root.find("solids")
        assert solids is not None
        rotations = [e for e in solids.iter() if e.tag == "firstrotation"]
        angles = {(r.get("x"), r.get("y")) for r in rotations}
        assert ("0", "1.5707963267948966") in angles
        assert ("1.5707963267948966", "0") in angles

    def test_complement_and_universe_fill(self) -> None:
        xml, drift = nucleide.mcnp.read_csg_to_gdml(str(FIXTURES / "deck_csg_complement.txt"))
        assert "subtraction" in xml
        assert any(d["action"] == "complement-expansion" for d in drift)
        xml, drift = nucleide.mcnp.read_csg_to_gdml(str(FIXTURES / "deck_csg_universe_fill.txt"))
        assert '<assembly name="asm1">' in xml
        assert any(d["action"] == "fill-applied" for d in drift)

    def test_rect_lattice_expands_placements(self) -> None:
        xml, drift = nucleide.mcnp.read_csg_to_gdml(str(FIXTURES / "deck_csg_lattice_rect.txt"))
        root = ET.fromstring(xml)
        structure = root.find("structure")
        assert structure is not None
        vol10 = next(v for v in structure.findall("volume") if v.get("name") == "vol10")
        placements = vol10.findall("physvol")
        assert len(placements) == 8
        positions = set()
        for put in placements:
            pos = put.find("position")
            assert pos is not None
            positions.add((pos.get("x"), pos.get("y"), pos.get("z")))
        assert positions == {
            ("10", "10", "10"),
            ("30", "10", "10"),
            ("10", "30", "10"),
            ("30", "30", "10"),
            ("10", "10", "30"),
            ("30", "10", "30"),
            ("10", "30", "30"),
            ("30", "30", "30"),
        }
        assert any(d["action"] == "lattice-expanded" and d["target"] == "10" for d in drift)

    def test_parse_text_matches_file(self) -> None:
        path = FIXTURES / "deck_csg_rpp.txt"
        file_xml, _ = nucleide.mcnp.read_csg_to_gdml(str(path))
        text_xml, _ = nucleide.mcnp.parse_csg_to_gdml(path.read_text())
        assert text_xml == file_xml

    def test_boundaries_rejected(self) -> None:
        with pytest.raises(ValueError, match="no gdml spelling"):
            nucleide.mcnp.parse_csg_to_gdml("msg\ntitle\n1 0 -1\n\n*1 so 10\n\n")
        with pytest.raises(ValueError, match="no gdml spelling"):
            nucleide.mcnp.parse_csg_to_gdml("msg\ntitle\n1 0 *-1\n\n1 so 10\n\n")
        with pytest.raises(ValueError, match="no gdml spelling"):
            nucleide.mcnp.parse_csg_to_gdml("msg\ntitle\n1 0 -1 2\n\n1 -2 pz 0\n2 pz 5\n\n")

    def test_out_of_scope_shared(self) -> None:
        with pytest.raises(ValueError, match="no v1 mapping"):
            nucleide.mcnp.parse_csg_to_gdml(
                "msg\ntitle\n1 1 -1.0 -1\n\n1 kz 0 0 0 1 1\n\nm1 1001 1.0\n"
            )
        with pytest.raises(ValueError, match="out of v1 scope"):
            nucleide.mcnp.parse_csg_to_gdml("msg\ntitle\n1 0 -1 u=-1\n\n1 so 10\n\n")
        with pytest.raises(ValueError, match="axis-aligned BOX"):
            nucleide.mcnp.parse_csg_to_gdml(
                "msg\ntitle\n1 1 -1.0 -1\n\n1 box 0 0 0 1 1 0 0 1 0 0 0 1\n\nm1 1001 1.0\n"
            )
