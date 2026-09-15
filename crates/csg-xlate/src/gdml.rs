//! GDML emission direction: MCNP CSG to Geant4 GDML (scoped v3).
//!
//! [`deck_csg_to_gdml`] is the fourth emitter direction over the shared v3
//! front-end (same scope checks, universe resolution, and drift-report
//! contract as the OpenMC/Serpent/PHITS directions). GDML is a
//! *solid*-based language: every MCNP cell becomes a named boolean solid in
//! `<solids>` plus a `<volume>` in `<structure>`, every universe becomes a
//! `<assembly>`, and the world volume is a `<volume>` referencing the shared
//! cutoff box.
//!
//! # Schema pinning
//!
//! The v1 element subset is pinned against the GDML schema **version 3.1.7**,
//! distributed in the Geant4 source tree at `source/persistency/gdml/schema/`
//! (tag `v11.4.2`, the latest stable release; the Geant4 Software License is
//! BSD-style and the schema is public). The GDML format is described by the
//! CHEP 2005 paper (CERN-CDS-1023367). The emitted subset uses, by file:
//!
//! - `gdml.xsd`: root `gdml` (fixed `version="3.1.7"`), `structure`,
//!   `volume`/`VolumeType` (`materialref`, `solidref`, `physvol`),
//!   `assembly`/`AssemblyVolumeType` (universe containers; the schema itself
//!   documents that the setup world volume cannot be an assembly),
//!   `physvol`/`SinglePlacementType` (`volumeref`, `position`, `rotation`),
//!   `setup` + `world`.
//! - `gdml_solids.xsd`: `solids`, `box`, `orb`, `tube`,
//!   `union`/`subtraction`/`intersection` (`BooleanSolidType` with `first`,
//!   `second`, `firstposition`, `firstrotation`).
//! - `gdml_materials.xsd`: `materials`, `material` (`D` + `atom` — the
//!   minimal validating content, emitted as caller-replaced stubs).
//! - `gdml_define.xsd`: `positionType` (unit default `mm`) and `rotationType`
//!   (unit default `radian`); both units are emitted explicitly.
//!
//! # Length units
//!
//! MCNP lengths are centimetres; GDML's schema default is millimetres. Every
//! emitted length — `<position>`/`<firstposition>` offsets and every solid
//! dimension (`box` `x`/`y`/`z`, `orb` `r`, `tube` `rmax`/`z`) — is converted
//! ×10 at the XML write, keeping the schema-default `unit="mm"` spelling
//! (angles stay radians). The internal solid/transform model stays in cm so
//! boolean compositions and the drift `halfspace-bounded` cutoff `L` remain
//! source-deck quantities; only the rendered numbers are mm.
//!
//! The schema declares no target namespace and unqualified element/attribute
//! form, so the document carries
//! `xsi:noNamespaceSchemaLocation="http://cern.ch/geant4/GDML/schema/gdml.xsd"`
//! (the location string Geant4 itself writes).
//!
//! Rotation convention (verified against `G4GDMLReadDefine.cc` of the pinned
//! tag): a `rotation` triplet `(x, y, z)` builds `R = Rz(z) * Ry(y) * Rx(x)`
//! applied to column vectors, angles in radians.
//!
//! # Per-kind solids mapping
//!
//! | MCNP kind | GDML v1 spelling | Lossy? |
//! |---|---|---|
//! | `PX`/`X`, `PY`/`Y`, `PZ`/`Z` | half-space `box` pair (`sl<n>p` = `x>d`, `sl<n>m` = `x<d`), center offset `d±L` on the axis | yes — the infinite plane is bounded by the per-deck cutoff `L` (`halfspace-bounded` drift) |
//! | `SO`/`S`/`SX`/`SY`/`SZ`/`SPH` | `orb` (`rmax=r`), translation offset for the center | no |
//! | `CX`/`CY`/`CZ` | `tube` (`rmax=r`, `z=2L`) at the origin; `CX` adds a `Ry(+90°)`, `CY` a `Rx(−90°)` firstrotation | yes — the infinite axis is bounded by `z=±L` (`halfspace-bounded` drift) |
//! | `RPP` | native closed `box` (exact dimensions), translation offset | no |
//! | `BOX` (axis-aligned edge vectors only) | native closed `box`, translation offset | no; canted edges are loud [`Error::MacrobodyOutOfScope`] |
//! | `RCC` (axis-aligned only) | named boolean `sl<n>` = `tube(z=\|h\|)` + two cap half-space boxes in the RCC local frame | no — the tube is finite and the cap boxes are cut by the tube radius |
//! | cones, quadrics, tori, three-point `P` | loud [`Error::UnsupportedSurface`] | — |
//! | other macrobodies (`REC`, `WED`, `RHP`, `HEX`, `TRC`, `ELL`, `ARB`, non-axis `RCC`/`BOX`) | loud [`Error::MacrobodyOutOfScope`] | — |
//!
//! Every boolean operand in GDML is a named-solid reference, so region folds
//! register one named solid per intermediate node (`csol<n>b<i>` cell chains,
//! `sl<n>b<i>` RCC chains, `dm<n>*` De Morgan rewrites).
//!
//! # Validation route
//!
//! Geant4 load stays OUT (disproportionate dependency for an emitter).
//! Structural assertions always run (Rust unit tests, Python tests, and the
//! `parsers_vs_refs.py` probe); XSD validation runs wherever `lxml` is
//! importable (it is preinstalled in the validation container) against the
//! runtime hash-pinned schema cache, and SKIPs loudly otherwise.

use std::collections::{BTreeMap, BTreeSet};

use nucleide_mcnp_io::cell::{CellCard, GeomExpr};
use nucleide_mcnp_io::problem::DeckProblem;
use nucleide_mcnp_io::surf::{SurfCard, SurfKind};
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, Event};
use quick_xml::writer::Writer;

use crate::{
    check_cell_params, check_data_cards, is_flat, resolve_universes, Ctx, DriftScope, DriftTable,
    Error, Result,
};

/// Schema version pinned for emission (Geant4 `v11.4.2` schema files).
const GDML_SCHEMA_VERSION: &str = "3.1.7";

/// Location hint Geant4 itself writes into exported documents.
const GDML_SCHEMA_LOCATION: &str = "http://cern.ch/geant4/GDML/schema/gdml.xsd";

/// 3x3 rotation matrix (row-major), built only from axis rotations here.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Rot([[f64; 3]; 3]);

impl Rot {
    const fn ident() -> Self {
        Rot([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    fn rotx(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Rot([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    }

    fn roty(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Rot([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    }

    fn rotz(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Rot([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    }

    fn mul(self, other: &Self) -> Self {
        let mut out = [[0.0; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, slot) in row.iter_mut().enumerate() {
                *slot = (0..3).map(|k| self.0[i][k] * other.0[k][j]).sum();
            }
        }
        Rot(out)
    }

    fn transpose(&self) -> Self {
        let mut out = [[0.0; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, slot) in row.iter_mut().enumerate() {
                *slot = self.0[j][i];
            }
        }
        Rot(out)
    }

    fn apply(&self, v: [f64; 3]) -> [f64; 3] {
        let mut out = [0.0; 3];
        for (slot, row) in out.iter_mut().zip(&self.0) {
            *slot = row.iter().zip(v).map(|(a, b)| a * b).sum();
        }
        out
    }

    /// True when numerically the identity (all angles here are multiples of
    /// 90 degrees, so the tolerance only absorbs floating-point noise).
    fn is_identity(&self) -> bool {
        self.0.iter().enumerate().all(|(i, row)| {
            row.iter()
                .enumerate()
                .all(|(j, v)| (v - if i == j { 1.0 } else { 0.0 }).abs() < 1.0e-9)
        })
    }

    /// Decompose into the GDML Euler triplet `(x, y, z)` meaning
    /// `R = Rz(z) * Ry(y) * Rx(x)` (the `G4GDMLReadDefine` convention).
    /// Single-axis rotations dominate v1; the gimbal branch covers the
    /// composed `RX^-1 * RY` products that appear when cylinders mix. The
    /// `+ 0.0` normalizes negative zero so emitted attributes stay clean.
    fn euler(&self) -> [f64; 3] {
        let r = &self.0;
        let y = (-r[2][0]).clamp(-1.0, 1.0).asin();
        if r[2][0].abs() < 1.0e-9 || (1.0 - r[2][0].abs()).abs() < 1.0e-9 {
            // Gimbal lock (or no y rotation): fold x into z.
            let x = if r[2][0].abs() < 1.0e-9 {
                (r[2][1]).atan2(r[2][2])
            } else {
                0.0
            };
            let z = if r[2][0].abs() < 1.0e-9 {
                (r[1][0]).atan2(r[0][0])
            } else {
                (-r[0][1]).atan2(r[1][1])
            };
            [x + 0.0, y + 0.0, z + 0.0]
        } else {
            let x = (r[2][1]).atan2(r[2][2]);
            let z = (r[1][0]).atan2(r[0][0]);
            [x + 0.0, y + 0.0, z + 0.0]
        }
    }
}

/// Rigid transform: world point = `r * p + t` (rotate, then translate, as in
/// `G4DisplacedSolid`).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Xf {
    r: Rot,
    t: [f64; 3],
}

impl Xf {
    const fn ident() -> Self {
        Xf {
            r: Rot::ident(),
            t: [0.0; 3],
        }
    }

    const fn trans(t: [f64; 3]) -> Self {
        Xf { r: Rot::ident(), t }
    }

    fn rot(r: Rot, t: [f64; 3]) -> Self {
        Xf { r, t }
    }

    /// Compose `self ∘ other` (apply `other` first).
    fn compose(&self, other: &Self) -> Self {
        Xf {
            r: self.r.mul(&other.r),
            t: add3(self.t, self.r.apply(other.t)),
        }
    }

    /// Relative transform `inv(other) ∘ self`, used to place the accumulated
    /// subtree (`self`) into a new boolean node anchored at `other`.
    fn rel_to(&self, other: &Self) -> Self {
        Xf {
            r: other.r.transpose().mul(&self.r),
            t: other.r.transpose().apply(sub3(self.t, other.t)),
        }
    }
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// One named solid definition for the `<solids>` section.
#[derive(Debug, Clone, PartialEq)]
enum SolidDef {
    /// `box`/`orb`/`tube` at the origin; placement rides on the reference.
    Prim {
        /// Element name (`box`, `orb`, `tube`).
        elem: &'static str,
        /// Ordered attribute values (x/y/z for box, r for orb,
        /// rmax/deltaphi/z for tube).
        vals: Vec<f64>,
    },
    /// A named boolean node: `op(first, second)` with the accumulated
    /// subtree displaced by `rel` relative to the second operand's frame.
    Bool {
        /// `union`, `subtraction`, or `intersection`.
        op: &'static str,
        /// First operand solid name (accumulated side).
        first: String,
        /// Second operand solid name (anchor side).
        second: String,
        /// Displacement of the first operand relative to the second.
        rel: Xf,
    },
}

/// Registry of named solids plus the per-deck half-space cutoff.
#[derive(Debug, Default)]
struct Solids {
    /// Half-space extent `L`: infinite primitives span `±L`, the world
    /// cutoff box spans `±L` per axis.
    l: f64,
    /// Solid name -> definition, in insertion order (BTreeMap keeps output
    /// deterministic).
    defs: BTreeMap<String, SolidDef>,
}

impl Solids {
    fn prim(&mut self, name: String, elem: &'static str, vals: Vec<f64>) -> String {
        self.defs
            .insert(name.clone(), SolidDef::Prim { elem, vals });
        name
    }

    /// Half-space box pair member for plane surface `n`; `positive` selects
    /// the `x>d` box (`p`) or the `x<d` box (`m`). The box is a cube of side
    /// `2L` at the origin; the reference transform centers it at `d±L`.
    fn hs_box(&mut self, surf: u32, positive: bool, axis: usize, d: f64) -> (String, Xf) {
        let name = format!("sl{}{}", surf, if positive { "p" } else { "m" });
        if !self.defs.contains_key(&name) {
            self.prim(
                name.clone(),
                "box",
                vec![2.0 * self.l, 2.0 * self.l, 2.0 * self.l],
            );
        }
        let mut t = [0.0; 3];
        t[axis] = if positive { d + self.l } else { d - self.l };
        (name, Xf::trans(t))
    }

    fn bool_node(
        &mut self,
        name: String,
        op: &'static str,
        first: String,
        second: String,
        rel: Xf,
    ) -> String {
        self.defs.insert(
            name.clone(),
            SolidDef::Bool {
                op,
                first,
                second,
                rel,
            },
        );
        name
    }
}

/// One resolved region piece: a closed solid available in a positive
/// (intersect/union) or negative (subtract/outside) role.
#[derive(Debug, Clone)]
enum Piece {
    /// Plane half-space carried as its two boxed senses so [`Piece::flip`]
    /// stays exact.
    Hs {
        /// Plane surface number.
        surf: u32,
        /// True = positive side (`x>d`).
        positive: bool,
        /// Axis index (0=x, 1=y, 2=z).
        axis: usize,
        /// Plane offset.
        d: f64,
    },
    /// Interior of a closed solid (named, since complements/unions may need
    /// to reference or negate it).
    In {
        /// Solid name.
        name: String,
        /// World transform of the solid.
        xf: Xf,
    },
    /// Exterior of a closed solid: usable standalone only as
    /// `subtraction(bigbox, solid)` or as a subtraction term.
    Out {
        /// Solid name.
        name: String,
        /// World transform of the solid.
        xf: Xf,
    },
}

impl Piece {
    /// Boolean complement of this piece (De Morgan leaf).
    fn flip(self) -> Piece {
        match self {
            Piece::Hs {
                surf,
                positive,
                axis,
                d,
            } => Piece::Hs {
                surf,
                positive: !positive,
                axis,
                d,
            },
            Piece::In { name, xf } => Piece::Out { name, xf },
            Piece::Out { name, xf } => Piece::In { name, xf },
        }
    }
}

/// A resolved region piece together with its registry-produced reference.
#[derive(Debug, Clone)]
struct Ref {
    /// Solid name to reference.
    name: String,
    /// World transform of the solid's frame.
    xf: Xf,
}

/// Resolve one piece to the reference used inside boolean folds.
fn lit_ref(solids: &mut Solids, piece: &Piece) -> Ref {
    match piece {
        Piece::Hs {
            surf,
            positive,
            axis,
            d,
        } => {
            let (name, xf) = solids.hs_box(*surf, *positive, *axis, *d);
            Ref { name, xf }
        }
        Piece::In { name, xf } | Piece::Out { name, xf } => Ref {
            name: name.clone(),
            xf: *xf,
        },
    }
}

/// Map one deck surface to its GDML piece pair (positive/negative sense).
///
/// Native closed solids (`RPP`, axis-aligned `BOX`/`RCC`, spheres) are exact;
/// planes and on-axis cylinders are half-space/tube spellings bounded by the
/// per-deck cutoff `L` (one `halfspace-bounded` deck note covers them).
/// Out-of-scope kinds reuse [`Error::UnsupportedSurface`] and
/// [`Error::MacrobodyOutOfScope`] with GDML-flavored details.
fn map_gdml_surface(solids: &mut Solids, card: &SurfCard) -> Result<GdmlSurf> {
    let c = &card.coeffs;
    let in_out = |name: String, xf: Xf| GdmlSurf {
        pos: Piece::Out {
            name: name.clone(),
            xf,
        },
        neg: Piece::In { name, xf },
    };
    let mapped = match card.kind {
        SurfKind::Px | SurfKind::X => GdmlSurf::hs(card.num, 0, c[0]),
        SurfKind::Py | SurfKind::Y => GdmlSurf::hs(card.num, 1, c[0]),
        SurfKind::Pz | SurfKind::Z => GdmlSurf::hs(card.num, 2, c[0]),
        SurfKind::So => {
            let name = orb(solids, card.num, [0.0; 3], c[0]);
            in_out(name, Xf::ident())
        }
        SurfKind::Sx => {
            let name = orb(solids, card.num, [c[0], 0.0, 0.0], c[1]);
            in_out(name, Xf::trans([c[0], 0.0, 0.0]))
        }
        SurfKind::Sy => {
            let name = orb(solids, card.num, [0.0, c[0], 0.0], c[1]);
            in_out(name, Xf::trans([0.0, c[0], 0.0]))
        }
        SurfKind::Sz => {
            let name = orb(solids, card.num, [0.0, 0.0, c[0]], c[1]);
            in_out(name, Xf::trans([0.0, 0.0, c[0]]))
        }
        SurfKind::S | SurfKind::Sph => {
            let center = [c[0], c[1], c[2]];
            let name = orb(solids, card.num, center, c[3]);
            in_out(name, Xf::trans(center))
        }
        SurfKind::Cx => {
            let name = tube(solids, card.num, c[0]);
            in_out(
                name,
                Xf::rot(Rot::roty(std::f64::consts::FRAC_PI_2), [0.0; 3]),
            )
        }
        SurfKind::Cy => {
            let name = tube(solids, card.num, c[0]);
            in_out(
                name,
                Xf::rot(Rot::rotx(-std::f64::consts::FRAC_PI_2), [0.0; 3]),
            )
        }
        SurfKind::Cz => {
            let name = tube(solids, card.num, c[0]);
            in_out(name, Xf::ident())
        }
        SurfKind::Rpp => {
            let spans = [c[1] - c[0], c[3] - c[2], c[5] - c[4]];
            let center = [
                (c[0] + c[1]) / 2.0,
                (c[2] + c[3]) / 2.0,
                (c[4] + c[5]) / 2.0,
            ];
            let name = solids.prim(
                format!("sl{}", card.num),
                "box",
                vec![spans[0], spans[1], spans[2]],
            );
            in_out(name, Xf::trans(center))
        }
        SurfKind::McBox => {
            let corner = [c[0], c[1], c[2]];
            let edges = [[c[3], c[4], c[5]], [c[6], c[7], c[8]], [c[9], c[10], c[11]]];
            let mut lo = corner;
            let mut hi = corner;
            for edge in edges {
                // v1 maps axis-aligned edge vectors only: each edge must be
                // parallel to one coordinate axis (a permutation covers the
                // general right-handed corner box).
                let nonzero: Vec<usize> = (0..3).filter(|&i| edge[i] != 0.0).collect();
                if nonzero.len() != 1 {
                    return Err(Error::MacrobodyOutOfScope {
                        surf: card.num,
                        kind: card.kind.keyword().to_string(),
                        detail: "gdml maps axis-aligned BOX edge vectors only \
                            (canted edges need arbitrary rotation composition)"
                            .to_string(),
                    });
                }
                let axis = nonzero[0];
                if edge[axis] < 0.0 {
                    lo[axis] += edge[axis];
                } else {
                    hi[axis] += edge[axis];
                }
            }
            let spans = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
            let center = [
                (lo[0] + hi[0]) / 2.0,
                (lo[1] + hi[1]) / 2.0,
                (lo[2] + hi[2]) / 2.0,
            ];
            let name = solids.prim(
                format!("sl{}", card.num),
                "box",
                vec![spans[0], spans[1], spans[2]],
            );
            in_out(name, Xf::trans(center))
        }
        SurfKind::Rcc => {
            let base = [c[0], c[1], c[2]];
            let axis = [c[3], c[4], c[5]];
            let radius = c[6];
            let nonzero: Vec<usize> = (0..3).filter(|&i| axis[i] != 0.0).collect();
            if nonzero.len() != 1 {
                return Err(Error::MacrobodyOutOfScope {
                    surf: card.num,
                    kind: card.kind.keyword().to_string(),
                    detail: "only RCCs aligned with x, y, or z emit in gdml \
                        (canted cylinders have no exact GDML spelling)"
                        .to_string(),
                });
            }
            let ax = nonzero[0];
            let h = axis[ax].abs();
            let sign = axis[ax].signum();
            let rot = match (ax, sign > 0.0) {
                (0, true) => Rot::roty(std::f64::consts::FRAC_PI_2),
                (0, false) => Rot::roty(-std::f64::consts::FRAC_PI_2),
                (1, true) => Rot::rotx(-std::f64::consts::FRAC_PI_2),
                (1, false) => Rot::rotx(std::f64::consts::FRAC_PI_2),
                (2, true) => Rot::ident(),
                _ => Rot::rotz(std::f64::consts::PI),
            };
            // Named boolean sl<n> in the RCC local frame: finite tube along
            // local +z (base at z=0, top at z=|h|) capped by two half-space
            // boxes. The cell reference transform composes the RCC world
            // placement with the fold output frame.
            let tube_name = solids.prim(
                format!("sl{}t", card.num),
                "tube",
                vec![radius, std::f64::consts::TAU, h],
            );
            let (cap_lo, xf_lo) = (
                format!("sl{}lo", card.num),
                Xf::trans([0.0, 0.0, -solids.l]),
            );
            let (cap_hi, xf_hi) = (
                format!("sl{}hi", card.num),
                Xf::trans([0.0, 0.0, h + solids.l]),
            );
            if !solids.defs.contains_key(&cap_lo) {
                solids.prim(
                    cap_lo.clone(),
                    "box",
                    vec![2.0 * solids.l, 2.0 * solids.l, 2.0 * solids.l],
                );
                solids.prim(
                    cap_hi.clone(),
                    "box",
                    vec![2.0 * solids.l, 2.0 * solids.l, 2.0 * solids.l],
                );
            }
            let tube_ref = Ref {
                name: tube_name,
                xf: Xf::trans([0.0, 0.0, h / 2.0]),
            };
            let node1 = fold_inter(
                solids,
                format!("sl{}b1", card.num),
                &tube_ref,
                &Ref {
                    name: cap_lo,
                    xf: xf_lo,
                },
            );
            let (root, out_xf) = fold_inter(
                solids,
                format!("sl{}", card.num),
                &node1.0,
                &Ref {
                    name: cap_hi,
                    xf: xf_hi,
                },
            );
            let world = Xf::rot(rot, base);
            GdmlSurf {
                pos: Piece::Out {
                    name: root.name.clone(),
                    xf: world.compose(&out_xf),
                },
                neg: Piece::In {
                    name: root.name,
                    xf: world.compose(&out_xf),
                },
            }
        }
        SurfKind::P => {
            return Err(Error::UnsupportedSurface {
                surf: card.num,
                kind: card.kind.keyword().to_string(),
                detail: "the 9-coefficient three-point plane has no exact GDML \
                spelling (half-space boxes cover PX/PY/PZ only)"
                    .to_string(),
            })
        }
        SurfKind::Kx | SurfKind::Ky | SurfKind::Kz => {
            return Err(Error::UnsupportedSurface {
                surf: card.num,
                kind: card.kind.keyword().to_string(),
                detail: "the 5-coefficient cone (apex, slope, sheet selector) has \
                no verified GDML cone/cone segment mapping"
                    .to_string(),
            })
        }
        SurfKind::Sq | SurfKind::Gq => {
            return Err(Error::UnsupportedSurface {
                surf: card.num,
                kind: card.kind.keyword().to_string(),
                detail: "the MCNP quadric coefficient ordering against GDML \
                quadric-family solids is unverified"
                    .to_string(),
            })
        }
        SurfKind::Tx | SurfKind::Ty | SurfKind::Tz => {
            return Err(Error::UnsupportedSurface {
                surf: card.num,
                kind: card.kind.keyword().to_string(),
                detail: "the 6-coefficient torus ordering against GDML torus \
                (rmin/rmax/r tor/r sect) is unverified"
                    .to_string(),
            })
        }
        SurfKind::Rec
        | SurfKind::Wed
        | SurfKind::Rhp
        | SurfKind::Hex
        | SurfKind::Trc
        | SurfKind::Ell
        | SurfKind::Arb => {
            return Err(Error::MacrobodyOutOfScope {
                surf: card.num,
                kind: card.kind.keyword().to_string(),
                detail: "gdml maps planes, spheres, on-axis cylinders, \
                SPH/RPP/axis-aligned RCC and axis-aligned BOX only"
                    .to_string(),
            })
        }
    };
    Ok(mapped)
}

/// The two senses of one mapped GDML surface.
struct GdmlSurf {
    /// Positive sense (`x>d`, exterior of a closed solid).
    pos: Piece,
    /// Negative sense (`x<d`, interior of a closed solid).
    neg: Piece,
}

impl GdmlSurf {
    fn hs(surf: u32, axis: usize, d: f64) -> Self {
        GdmlSurf {
            pos: Piece::Hs {
                surf,
                positive: true,
                axis,
                d,
            },
            neg: Piece::Hs {
                surf,
                positive: false,
                axis,
                d,
            },
        }
    }

    fn pick(&self, sense_positive: bool) -> Piece {
        if sense_positive {
            self.pos.clone()
        } else {
            self.neg.clone()
        }
    }
}

/// Register an `orb` primitive (origin-centered).
fn orb(solids: &mut Solids, num: u32, _center: [f64; 3], r: f64) -> String {
    let name = format!("sl{num}");
    if !solids.defs.contains_key(&name) {
        solids.prim(name.clone(), "orb", vec![r]);
    }
    name
}

/// Register a `tube` primitive (z-axis, origin-centered, bounded by ±L).
fn tube(solids: &mut Solids, num: u32, r: f64) -> String {
    let name = format!("sl{num}");
    if !solids.defs.contains_key(&name) {
        solids.prim(
            name.clone(),
            "tube",
            vec![r, std::f64::consts::TAU, 2.0 * solids.l],
        );
    }
    name
}

/// Intersection fold step: registers `name` = `first ∩ second` with `first`
/// displaced by `rel` into `second`'s frame; returns the node and the output
/// frame (`second`'s world transform).
fn fold_inter(solids: &mut Solids, name: String, acc: &Ref, next: &Ref) -> (Ref, Xf) {
    let rel = acc.xf.rel_to(&next.xf);
    let node = solids.bool_node(
        name,
        "intersection",
        acc.name.clone(),
        next.name.clone(),
        rel,
    );
    (
        Ref {
            name: node,
            xf: next.xf,
        },
        next.xf,
    )
}

/// GDML region resolution state: deck views plus the solids registry.
struct Gdml<'a> {
    cells: BTreeMap<u32, &'a CellCard>,
    maps: BTreeMap<u32, GdmlSurf>,
    solids: Solids,
    /// Drift entries produced during region resolution (merged into the
    /// crate drift table after the region pass, keeping surfaces → cells →
    /// deck ordering).
    notes: Vec<crate::DriftEntry>,
}

impl<'a> Gdml<'a> {
    /// Resolve one surface literal (sense `surf > 0` = positive side).
    fn lit(&mut self, cell: u32, surf: i32, reflecting: bool) -> Result<Piece> {
        if reflecting {
            return Err(Error::GdmlBoundaryOutOfScope {
                surf: surf.unsigned_abs(),
                detail: format!("cell {cell} `*` marker has no gdml spelling"),
            });
        }
        let num = surf.unsigned_abs();
        let mapped = self
            .maps
            .get(&num)
            .ok_or(Error::UnknownSurface { cell, surf: num })?;
        Ok(mapped.pick(surf > 0))
    }

    /// De Morgan inline of `#n` over a flat region (same contract as the
    /// OpenMC direction: single half-space or intersection of half-spaces).
    fn complement(&mut self, cell: u32, target: i32) -> Result<Piece> {
        if target <= 0 {
            return Err(Error::UnknownCell { cell, target });
        }
        let target_num = target as u32;
        let referenced = self
            .cells
            .get(&target_num)
            .copied()
            .ok_or(Error::UnknownCell { cell, target })?;
        if !is_flat(&referenced.geom) {
            return Err(Error::ComplementTooComplex { cell, target });
        }
        self.notes.push(crate::DriftEntry {
            scope: DriftScope::Cell,
            target: cell,
            action: "complement-expansion".to_string(),
            reason: format!("#{target} inlined as the negated region of cell {target}"),
        });
        let mut lits: Vec<(i32, bool)> = Vec::new();
        fn walk(expr: &GeomExpr, out: &mut Vec<(i32, bool)>) {
            match expr {
                GeomExpr::HalfSpace(h) => out.push((h.surf, h.reflecting)),
                GeomExpr::Intersect(parts) => parts.iter().for_each(|p| walk(p, out)),
                GeomExpr::Union(..) | GeomExpr::Complement(..) => {
                    unreachable!("flat regions hold no unions or complements")
                }
            }
        }
        walk(&referenced.geom, &mut lits);
        let mut acc: Option<Piece> = None;
        for (surf, reflecting) in lits {
            // Negating the sense before resolving is the De Morgan leaf:
            // `lit(-surf)` is already the complement piece (interior becomes
            // exterior, one boxed half-space sense becomes the other).
            let negated = self.lit(cell, -surf, reflecting)?;
            acc = Some(match acc {
                None => negated,
                Some(prev) => union_pieces(self, cell, prev, negated)?,
            });
        }
        Ok(acc.expect("flat regions hold at least one literal"))
    }

    /// Resolve one geometry expression to a single piece (unions folded).
    fn resolve(&mut self, cell: u32, expr: &GeomExpr) -> Result<Piece> {
        match expr {
            GeomExpr::HalfSpace(h) => self.lit(cell, h.surf, h.reflecting),
            GeomExpr::Intersect(parts) => {
                let mut ins: Vec<Piece> = Vec::with_capacity(parts.len());
                let mut outs: Vec<Piece> = Vec::new();
                for part in parts {
                    match self.resolve(cell, part)? {
                        Piece::Out { name, xf } => outs.push(Piece::Out { name, xf }),
                        other => ins.push(other),
                    }
                }
                if ins.is_empty() && outs.is_empty() {
                    // Empty region: all space becomes the cutoff box.
                    let name = self.bigbox();
                    return Ok(Piece::In {
                        name,
                        xf: Xf::ident(),
                    });
                }
                let mut acc: Option<Ref> = None;
                if !ins.is_empty() {
                    let first = lit_ref(&mut self.solids, &ins[0]);
                    acc = Some(ins[1..].iter().try_fold(first, |a, p| {
                        let r = lit_ref(&mut self.solids, p);
                        let name = node_name(&self.solids, "csol", cell);
                        let (node, _) = fold_inter(&mut self.solids, name, &a, &r);
                        Ok::<Ref, Error>(node)
                    })?);
                }
                for out in &outs {
                    let o = lit_ref(&mut self.solids, out);
                    let base = acc.unwrap_or_else(|| Ref {
                        name: self.bigbox(),
                        xf: Xf::ident(),
                    });
                    let name = node_name(&self.solids, "csol", cell);
                    let (node, _) = fold_sub(&mut self.solids, name, &base, &o);
                    acc = Some(node);
                }
                let r = acc.expect("non-empty regions fold to something");
                Ok(Piece::In {
                    name: r.name,
                    xf: r.xf,
                })
            }
            GeomExpr::Union(a, b) => {
                let pa = self.resolve(cell, a)?;
                let pb = self.resolve(cell, b)?;
                union_pieces(self, cell, pa, pb)
            }
            GeomExpr::Complement(inner) => match inner.as_ref() {
                GeomExpr::HalfSpace(h) => self.complement(cell, h.surf),
                _ => Err(Error::ComplementTooComplex { cell, target: -1 }),
            },
        }
    }

    /// Register the shared cutoff box and return its name.
    fn bigbox(&mut self) -> String {
        if !self.solids.defs.contains_key("bigbox") {
            let l = self.solids.l;
            self.solids
                .prim("bigbox".to_string(), "box", vec![2.0 * l, 2.0 * l, 2.0 * l]);
        }
        "bigbox".to_string()
    }
}

/// True when the piece participates as a plain solid (intersection or union
/// member) rather than an exterior.
fn is_solid(piece: &Piece) -> bool {
    !matches!(piece, Piece::Out { .. })
}

/// Unique fold-node name derived from the registry size (deterministic for a
/// given deck because folds register immediately).
fn node_name(solids: &Solids, base: &str, cell: u32) -> String {
    format!("{base}{cell}b{}", solids.defs.len())
}

/// Subtraction fold step: registers `name` = `first − second`.
fn fold_sub(solids: &mut Solids, name: String, acc: &Ref, next: &Ref) -> (Ref, Xf) {
    let rel = acc.xf.rel_to(&next.xf);
    let node = solids.bool_node(
        name,
        "subtraction",
        acc.name.clone(),
        next.name.clone(),
        rel,
    );
    (
        Ref {
            name: node,
            xf: next.xf,
        },
        next.xf,
    )
}

/// Union fold step: registers `name` = `first ∪ second`.
fn fold_union(solids: &mut Solids, name: String, acc: &Ref, next: &Ref) -> (Ref, Xf) {
    let rel = acc.xf.rel_to(&next.xf);
    let node = solids.bool_node(name, "union", acc.name.clone(), next.name.clone(), rel);
    (
        Ref {
            name: node,
            xf: next.xf,
        },
        next.xf,
    )
}

/// Reference to the boolean complement of `piece`, registering a
/// `subtraction(bigbox, …)` node when the piece is a plain solid.
fn neg_ref(g: &mut Gdml<'_>, cell: u32, piece: Piece) -> Ref {
    match piece {
        Piece::Hs { .. } => {
            let flipped = piece.flip();
            lit_ref(&mut g.solids, &flipped)
        }
        Piece::In { name, xf } => {
            let big = Ref {
                name: g.bigbox(),
                xf: Xf::ident(),
            };
            let target = Ref { name, xf };
            let name = node_name(&g.solids, "dm", cell);
            let (node, _) = fold_sub(&mut g.solids, name, &big, &target);
            node
        }
        Piece::Out { name, xf } => Ref { name, xf },
    }
}

/// Fold two resolved pieces under a union. Unions mixing an exterior with a
/// solid rewrite through De Morgan inside the cutoff box:
/// `a ∪ b = ¬(¬a ∩ ¬b) = subtraction(bigbox, intersection(¬a, ¬b))`, which is
/// exact within the deck cutoff.
fn union_pieces(g: &mut Gdml<'_>, cell: u32, a: Piece, b: Piece) -> Result<Piece> {
    if is_solid(&a) && is_solid(&b) {
        let ra = lit_ref(&mut g.solids, &a);
        let rb = lit_ref(&mut g.solids, &b);
        let name = node_name(&g.solids, "csol", cell);
        let (node, _) = fold_union(&mut g.solids, name, &ra, &rb);
        return Ok(Piece::In {
            name: node.name,
            xf: node.xf,
        });
    }
    let na = neg_ref(g, cell, a);
    let nb = neg_ref(g, cell, b);
    let name = node_name(&g.solids, "dm", cell);
    let (inner, _) = fold_inter(&mut g.solids, name, &na, &nb);
    let big = Ref {
        name: g.bigbox(),
        xf: Xf::ident(),
    };
    let name = node_name(&g.solids, "dm", cell);
    let (root, _) = fold_sub(&mut g.solids, name, &big, &inner);
    Ok(Piece::In {
        name: root.name,
        xf: root.xf,
    })
}

/// Format one f64 the shortest round-trip way (matches the other directions).
fn num(v: f64) -> String {
    v.to_string()
}

/// Scale one MCNP centimetre length to GDML millimetres (schema default).
fn mm(v: f64) -> f64 {
    10.0 * v
}

/// Scale a cm translation triplet to mm.
fn mm3(t: [f64; 3]) -> [f64; 3] {
    [mm(t[0]), mm(t[1]), mm(t[2])]
}

/// Push `x`/`y`/`z` attributes onto a start element.
fn push_xyz(elem: &mut BytesStart<'_>, t: [f64; 3]) {
    for (k, v) in [("x", t[0]), ("y", t[1]), ("z", t[2])] {
        let s = num(v);
        elem.push_attribute((k, s.as_str()));
    }
}

/// Write a `<position>`/`<rotation>` element for a transform, skipping each
/// when identity (both are optional in the schema but take a required
/// document-unique `name` attribute, so each emitted element gets one).
fn write_xf(
    writer: &mut Writer<Vec<u8>>,
    xf: &Xf,
    seq: &mut u32,
) -> std::result::Result<(), quick_xml::Error> {
    if !xf.t.iter().all(|v| v.abs() < 1.0e-12) {
        let mut elem = BytesStart::new("position");
        let name = format!("pos{seq}");
        *seq += 1;
        elem.push_attribute(("name", name.as_str()));
        push_xyz(&mut elem, mm3(xf.t));
        elem.push_attribute(("unit", "mm"));
        writer.write_event(Event::Empty(elem))?;
    }
    if !xf.r.is_identity() {
        let angles = xf.r.euler();
        let mut elem = BytesStart::new("rotation");
        let name = format!("rot{seq}");
        *seq += 1;
        elem.push_attribute(("name", name.as_str()));
        push_xyz(&mut elem, angles);
        elem.push_attribute(("unit", "radian"));
        writer.write_event(Event::Empty(elem))?;
    }
    Ok(())
}

/// Write one `<physvol>` placing `vol` at `xf`.
fn write_physvol(
    writer: &mut Writer<Vec<u8>>,
    vol: &str,
    xf: &Xf,
    seq: &mut u32,
) -> std::result::Result<(), quick_xml::Error> {
    writer.write_event(Event::Start(BytesStart::new("physvol")))?;
    let mut elem = BytesStart::new("volumeref");
    elem.push_attribute(("ref", vol));
    writer.write_event(Event::Empty(elem))?;
    write_xf(writer, xf, seq)?;
    writer.write_event(Event::End(BytesEnd::new("physvol")))?;
    Ok(())
}

/// Write the `<solids>` section from the registry.
fn write_solids(
    writer: &mut Writer<Vec<u8>>,
    solids: &Solids,
    seq: &mut u32,
) -> std::result::Result<(), quick_xml::Error> {
    writer.write_event(Event::Start(BytesStart::new("solids")))?;
    for (name, def) in &solids.defs {
        match def {
            SolidDef::Prim { elem, vals } => {
                let mut elem_start = BytesStart::new(*elem);
                elem_start.push_attribute(("name", name.as_str()));
                match *elem {
                    "box" => {
                        for (k, v) in ["x", "y", "z"].iter().zip(vals) {
                            let s = num(mm(*v));
                            elem_start.push_attribute((*k, s.as_str()));
                        }
                    }
                    "orb" => {
                        let s = num(mm(vals[0]));
                        elem_start.push_attribute(("r", s.as_str()));
                    }
                    _ => {
                        // tube: rmax, deltaphi, z (rmin/startphi default).
                        // `rmax`/`z` are lengths (cm -> mm); `deltaphi` is an
                        // angle and passes through unchanged.
                        for (k, v) in ["rmax", "deltaphi", "z"].iter().zip(vals) {
                            let s = num(if *k == "deltaphi" { *v } else { mm(*v) });
                            elem_start.push_attribute((*k, s.as_str()));
                        }
                    }
                }
                writer.write_event(Event::Empty(elem_start))?;
            }
            SolidDef::Bool {
                op,
                first,
                second,
                rel,
            } => {
                let mut node_start = BytesStart::new(*op);
                node_start.push_attribute(("name", name.as_str()));
                writer.write_event(Event::Start(node_start))?;
                let mut f = BytesStart::new("first");
                f.push_attribute(("ref", first.as_str()));
                writer.write_event(Event::Empty(f))?;
                let mut s = BytesStart::new("second");
                s.push_attribute(("ref", second.as_str()));
                writer.write_event(Event::Empty(s))?;
                if !rel.t.iter().all(|v| v.abs() < 1.0e-12) {
                    let mut p = BytesStart::new("firstposition");
                    let name = format!("fp{seq}");
                    *seq += 1;
                    p.push_attribute(("name", name.as_str()));
                    push_xyz(&mut p, mm3(rel.t));
                    p.push_attribute(("unit", "mm"));
                    writer.write_event(Event::Empty(p))?;
                }
                if !rel.r.is_identity() {
                    let mut r = BytesStart::new("firstrotation");
                    let name = format!("fr{seq}");
                    *seq += 1;
                    r.push_attribute(("name", name.as_str()));
                    push_xyz(&mut r, rel.r.euler());
                    r.push_attribute(("unit", "radian"));
                    writer.write_event(Event::Empty(r))?;
                }
                writer.write_event(Event::End(BytesEnd::new(*op)))?;
            }
        }
    }
    writer.write_event(Event::End(BytesEnd::new("solids")))?;
    Ok(())
}

/// Derive the per-deck half-space cutoff: ten times the largest absolute
/// coefficient over every deck surface (floor 1.0 so tiny decks keep
/// generous half-space boxes). Unmapped kinds error out of the surface pass
/// anyway, so every coefficient counted here belongs to a mapped solid.
fn cutoff(deck: &DeckProblem) -> f64 {
    let mut extent = 1.0_f64;
    for card in &deck.surfs {
        for v in &card.coeffs {
            extent = extent.max(v.abs());
        }
    }
    10.0 * extent
}

/// Translate a parsed MCNP deck to a GDML document plus drift.
///
/// Same v3 scope as the other directions (surfaces, cells, simple nested
/// universes, rectangular `LAT=1` lattices, material stub), expressed in
/// GDML's solid-based model:
///
/// - Every MCNP cell becomes a named boolean solid (or a direct primitive
///   reference) plus one `<volume>`; a universe becomes an `<assembly>` of
///   its cell volumes, the deck becomes a `world` `<volume>` whose solid is
///   the shared cutoff box, and `<setup>` points at it.
/// - `RPP` and axis-aligned `BOX`/`RCC` map to native closed solids (RCC as
///   a finite tube plus two cap half-space boxes); spheres map to `orb`;
///   axis planes and infinite cylinder axes are bounded by the per-deck
///   cutoff `L` (recorded once as a deck-scoped `halfspace-bounded` drift
///   note, since Geant4 has no infinite solids).
/// - `#n` complements inline by De Morgan over flat regions exactly like the
///   OpenMC direction; unions mixing an exterior rewrite through
///   `subtraction(bigbox, …)`, exact within the cutoff.
/// - Rectangular lattices expand to one `<physvol>` placement per element of
///   the element-universe assembly (`lattice-expanded` drift note per
///   lattice cell; `lattice-emitted` comes from the shared universe
///   resolution).
/// - Materials emit as `mat_<n>` stubs (`<D>`/`<atom>` placeholders the
///   caller replaces; one deck-scoped `material-stub` note lists them).
///
/// Reflecting and periodic boundaries stay loud ([`Error::GdmlBoundaryOutOfScope`]):
/// Geant4 expresses boundaries through wrapper code, not GDML.
pub fn deck_csg_to_gdml(deck: &DeckProblem) -> Result<(String, DriftTable)> {
    deck.validate()?;
    let mut ctx = Ctx {
        cells: deck.cells.iter().map(|c| (c.num, c)).collect(),
        surfs: deck.surfs.iter().map(|s| (s.num, s)).collect(),
        macros: BTreeMap::new(),
        next_id: deck.surfs.iter().map(|s| s.num).max().unwrap_or(0) + 1,
        forced_reflective: BTreeSet::new(),
        drift: DriftTable::default(),
    };
    for cell in &deck.cells {
        check_cell_params(&mut ctx, cell)?;
    }
    check_data_cards(&mut ctx, deck)?;
    let (cell_universe, cell_fill, cell_lattice) = resolve_universes(&mut ctx, deck)?;

    // Boundary pass first: GDML has no per-surface boundary spelling.
    for card in &deck.surfs {
        if let Some(tr) = card.transform {
            return Err(Error::TransformOutOfScope {
                detail: format!("surface {} links transform {tr}", card.num),
            });
        }
        if card.reflecting {
            return Err(Error::GdmlBoundaryOutOfScope {
                surf: card.num,
                detail: "reflective surface card has no gdml spelling".to_string(),
            });
        }
        if card.periodic.is_some() {
            return Err(Error::GdmlBoundaryOutOfScope {
                surf: card.num,
                detail: "periodic surface pointer has no gdml spelling".to_string(),
            });
        }
    }

    // Surface pass: map every deck surface (loud on unmapped kinds even
    // when unreferenced, matching the other directions).
    let mut g = Gdml {
        cells: deck.cells.iter().map(|c| (c.num, c)).collect(),
        maps: BTreeMap::new(),
        solids: Solids::default(),
        notes: Vec::new(),
    };
    // The cutoff must be in place before any surface maps: half-space boxes
    // and tubes take their spans from it at registration time.
    g.solids.l = cutoff(deck);
    // The world cutoff box is referenced by every translation's world
    // volume, so it must exist before the <solids> section is written.
    g.bigbox();
    for card in &deck.surfs {
        let mapped = map_gdml_surface(&mut g.solids, card)?;
        g.maps.insert(card.num, mapped);
    }
    ctx.drift.entries.push(crate::DriftEntry {
        scope: DriftScope::Deck,
        target: 0,
        action: "halfspace-bounded".to_string(),
        reason: format!(
            "infinite half-spaces bounded by |coord| <= {} mm (L = {} cm in the \
             source deck; Geant4 has no infinite solids; cell shapes exact \
             within the cutoff)",
            num(mm(g.solids.l)),
            num(g.solids.l)
        ),
    });
    for card in &deck.surfs {
        if matches!(card.kind, SurfKind::Rcc) {
            ctx.drift.entries.push(crate::DriftEntry {
                scope: DriftScope::Surface,
                target: card.num,
                action: "macrobody-expansion".to_string(),
                reason: format!(
                    "RCC {} emitted as named boolean solid sl{} (finite tube plus cap boxes)",
                    card.num, card.num
                ),
            });
        }
    }

    // Region pass: resolve each cell's geometry to one named solid plus the
    // world transform of its frame.
    let mut cell_refs: BTreeMap<u32, Ref> = BTreeMap::new();
    for cell in &deck.cells {
        let piece = g.resolve(cell.num, &cell.geom)?;
        let r#ref = match &piece {
            Piece::Hs { .. } => lit_ref(&mut g.solids, &piece),
            Piece::In { name, xf } => Ref {
                name: name.clone(),
                xf: *xf,
            },
            Piece::Out { name, xf } => {
                // Exterior standalone: everything in the cutoff box outside
                // the closed solid.
                let big = Ref {
                    name: g.bigbox(),
                    xf: Xf::ident(),
                };
                let target = Ref {
                    name: name.clone(),
                    xf: *xf,
                };
                let (node, _) =
                    fold_sub(&mut g.solids, format!("csol{}o", cell.num), &big, &target);
                node
            }
        };
        cell_refs.insert(cell.num, r#ref);
    }
    ctx.drift.entries.append(&mut g.notes);

    // Structure: one <volume> per cell, one <assembly> per universe >= 1,
    // and the world volume holding every universe-0 cell.
    let mut material_ids: BTreeSet<u32> = BTreeSet::new();
    material_ids.insert(0);
    let mut universe_members: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for cell in &deck.cells {
        let universe = cell_universe.get(&cell.num).copied().unwrap_or(0);
        universe_members.entry(universe).or_default().push(cell.num);
        if !cell_fill.contains_key(&cell.num) && !cell_lattice.contains_key(&cell.num) {
            material_ids.insert(cell.mat);
        }
    }

    let asm_name = |u: u32| format!("asm{u}");
    let vol_name = |n: u32| format!("vol{n}");

    // Lattice expansion: one physvol per element of the element universe.
    let mut placements: BTreeMap<u32, Vec<(u32, [f64; 3])>> = BTreeMap::new();
    for lat in cell_lattice.values() {
        let mut put = Vec::with_capacity(lat.counts.iter().product());
        for k in 0..lat.counts[2] {
            for j in 0..lat.counts[1] {
                for i in 0..lat.counts[0] {
                    let idx = [i, j, k];
                    let u = lat.universes[(k * lat.counts[1] + j) * lat.counts[0] + i];
                    let center =
                        [0, 1, 2].map(|a| lat.lower[a] + (idx[a] as f64 + 0.5) * lat.pitch[a]);
                    put.push((u, center));
                }
            }
        }
        placements.insert(lat.cell, put);
        let count: usize = lat.counts.iter().product();
        ctx.drift.entries.push(crate::DriftEntry {
            scope: DriftScope::Cell,
            target: lat.cell,
            action: "lattice-expanded".to_string(),
            reason: format!(
                "cell {} rectangular lattice expanded to {count} physvol \
                 placements of element assemblies",
                lat.cell
            ),
        });
    }
    ctx.drift.entries.push(crate::DriftEntry {
        scope: DriftScope::Deck,
        target: 0,
        action: "material-stub".to_string(),
        reason: format!(
            "material stubs ({}) emitted with placeholder D/atom values; \
             replace them with real definitions",
            material_ids
                .iter()
                .map(|m| format!("mat{m}"))
                .collect::<Vec<_>>()
                .join(" ")
        ),
    });

    let mut writer = Writer::new_with_indent(Vec::new(), b' ', 2);
    let decl = BytesDecl::new("1.0", Some("utf-8"), None);
    writer.write_event(Event::Decl(decl)).map_err(xml_error)?;
    writer
        .write_event(Event::Comment(quick_xml::events::BytesText::new(
            " GDML emitted from MCNP CSG by nucleide-csg-xlate. Scoped: \
             surfaces, cells, nested universes, rectangular LAT=1 lattices \
             (expanded placements). Replace material stubs; see the drift \
             report for the half-space cutoff and every approximation. ",
        )))
        .map_err(xml_error)?;
    let mut root = BytesStart::new("gdml");
    root.push_attribute(("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance"));
    root.push_attribute(("xsi:noNamespaceSchemaLocation", GDML_SCHEMA_LOCATION));
    root.push_attribute(("version", GDML_SCHEMA_VERSION));
    writer.write_event(Event::Start(root)).map_err(xml_error)?;
    writer
        .write_event(Event::Empty(BytesStart::new("define")))
        .map_err(xml_error)?;

    // <materials>: minimal validating stubs (D + atom required choices).
    writer
        .write_event(Event::Start(BytesStart::new("materials")))
        .map_err(xml_error)?;
    for m in &material_ids {
        let mut mat = BytesStart::new("material");
        let name = format!("mat{m}");
        mat.push_attribute(("name", name.as_str()));
        writer.write_event(Event::Start(mat)).map_err(xml_error)?;
        let mut d = BytesStart::new("D");
        d.push_attribute(("value", "1"));
        d.push_attribute(("unit", "g/cm3"));
        writer.write_event(Event::Empty(d)).map_err(xml_error)?;
        let mut atom = BytesStart::new("atom");
        atom.push_attribute(("value", "1"));
        atom.push_attribute(("unit", "g/mole"));
        writer.write_event(Event::Empty(atom)).map_err(xml_error)?;
        writer
            .write_event(Event::End(BytesEnd::new("material")))
            .map_err(xml_error)?;
    }
    writer
        .write_event(Event::End(BytesEnd::new("materials")))
        .map_err(xml_error)?;

    let mut seq = 0_u32;
    write_solids(&mut writer, &g.solids, &mut seq).map_err(xml_error)?;

    // <structure>: assemblies (universes >= 1), cell volumes, world volume.
    writer
        .write_event(Event::Start(BytesStart::new("structure")))
        .map_err(xml_error)?;
    for (universe, members) in &universe_members {
        if *universe == 0 {
            continue;
        }
        let mut asm = BytesStart::new("assembly");
        let name = asm_name(*universe);
        asm.push_attribute(("name", name.as_str()));
        writer.write_event(Event::Start(asm)).map_err(xml_error)?;
        for member in members {
            write_physvol(
                &mut writer,
                &vol_name(*member),
                &cell_refs[member].xf,
                &mut seq,
            )
            .map_err(xml_error)?;
        }
        writer
            .write_event(Event::End(BytesEnd::new("assembly")))
            .map_err(xml_error)?;
    }
    for cell in &deck.cells {
        let mut vol = BytesStart::new("volume");
        let name = vol_name(cell.num);
        vol.push_attribute(("name", name.as_str()));
        writer.write_event(Event::Start(vol)).map_err(xml_error)?;
        let mut mat_ref = BytesStart::new("materialref");
        let mat_name = if cell_fill.contains_key(&cell.num) || cell_lattice.contains_key(&cell.num)
        {
            "mat0".to_string()
        } else {
            format!("mat{}", cell.mat)
        };
        mat_ref.push_attribute(("ref", mat_name.as_str()));
        writer
            .write_event(Event::Empty(mat_ref))
            .map_err(xml_error)?;
        let mut solid_ref = BytesStart::new("solidref");
        solid_ref.push_attribute(("ref", cell_refs[&cell.num].name.as_str()));
        writer
            .write_event(Event::Empty(solid_ref))
            .map_err(xml_error)?;
        if let Some(fill) = cell_fill.get(&cell.num) {
            write_physvol(&mut writer, &asm_name(*fill), &Xf::ident(), &mut seq)
                .map_err(xml_error)?;
        }
        if let (Some(_), Some(puts)) = (cell_lattice.get(&cell.num), placements.get(&cell.num)) {
            for (u, center) in puts {
                write_physvol(&mut writer, &asm_name(*u), &Xf::trans(*center), &mut seq)
                    .map_err(xml_error)?;
            }
        }
        writer
            .write_event(Event::End(BytesEnd::new("volume")))
            .map_err(xml_error)?;
    }
    {
        let mut world = BytesStart::new("volume");
        world.push_attribute(("name", "world"));
        writer.write_event(Event::Start(world)).map_err(xml_error)?;
        let mut mat_ref = BytesStart::new("materialref");
        mat_ref.push_attribute(("ref", "mat0"));
        writer
            .write_event(Event::Empty(mat_ref))
            .map_err(xml_error)?;
        let mut solid_ref = BytesStart::new("solidref");
        solid_ref.push_attribute(("ref", g.bigbox().as_str()));
        writer
            .write_event(Event::Empty(solid_ref))
            .map_err(xml_error)?;
        for member in universe_members.get(&0).map(Vec::as_slice).unwrap_or(&[]) {
            write_physvol(
                &mut writer,
                &vol_name(*member),
                &cell_refs[member].xf,
                &mut seq,
            )
            .map_err(xml_error)?;
        }
        writer
            .write_event(Event::End(BytesEnd::new("volume")))
            .map_err(xml_error)?;
    }
    writer
        .write_event(Event::End(BytesEnd::new("structure")))
        .map_err(xml_error)?;

    let mut setup = BytesStart::new("setup");
    setup.push_attribute(("name", "default"));
    setup.push_attribute(("version", "1.0"));
    writer.write_event(Event::Start(setup)).map_err(xml_error)?;
    let mut world_ref = BytesStart::new("world");
    world_ref.push_attribute(("ref", "world"));
    writer
        .write_event(Event::Empty(world_ref))
        .map_err(xml_error)?;
    writer
        .write_event(Event::End(BytesEnd::new("setup")))
        .map_err(xml_error)?;
    writer
        .write_event(Event::End(BytesEnd::new("gdml")))
        .map_err(xml_error)?;
    let xml = String::from_utf8(writer.into_inner()).map_err(xml_error)?;
    Ok((xml, ctx.drift))
}

fn xml_error(e: impl std::fmt::Display) -> Error {
    Error::Xml {
        detail: e.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nucleide_mcnp_io::problem::parse_deck;

    fn translate(text: &str) -> Result<(String, DriftTable)> {
        deck_csg_to_gdml(&parse_deck(text).unwrap())
    }

    fn deck_text(cells: &str, surfs: &str, data: &str) -> String {
        format!("msg\ntitle\n{cells}\n\n{surfs}\n\n{data}\n")
    }

    #[test]
    fn euler_round_trips_axis_rotations() {
        for rot in [
            Rot::ident(),
            Rot::rotx(std::f64::consts::FRAC_PI_2),
            Rot::rotx(-std::f64::consts::FRAC_PI_2),
            Rot::roty(std::f64::consts::FRAC_PI_2),
            Rot::roty(-std::f64::consts::FRAC_PI_2),
            Rot::rotz(std::f64::consts::PI),
            Rot::rotx(-std::f64::consts::FRAC_PI_2).mul(&Rot::roty(std::f64::consts::FRAC_PI_2)),
        ] {
            let [x, y, z] = rot.euler();
            let rebuilt = Rot::rotz(z).mul(&Rot::roty(y)).mul(&Rot::rotx(x));
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        (rebuilt.0[i][j] - rot.0[i][j]).abs() < 1.0e-12,
                        "euler round trip failed: {rot:?} vs {rebuilt:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn sphere_in_plane_box_translates() {
        let (xml, drift) = translate(&deck_text(
            "1 1 -1.0 -1\n2 0 1 2 -3 4 -5 6 -7",
            "1 so 5.0\n2 px -10.0\n3 px 10.0\n4 py -10.0\n5 py 10.0\n6 pz -10.0\n7 pz 10.0",
            "m1 92235 1.0",
        ))
        .unwrap();
        assert!(xml.contains("version=\"3.1.7\""));
        assert!(xml.contains(
            "xsi:noNamespaceSchemaLocation=\"http://cern.ch/geant4/GDML/schema/gdml.xsd\""
        ));
        assert!(xml.contains("<define/>"));
        // Sphere: origin-centered orb referenced directly by its volume.
        // 5 cm -> 50 mm (MCNP cm converts x10 to GDML mm).
        assert!(xml.contains("<orb name=\"sl1\" r=\"50\"/>"));
        assert!(xml.contains("<volume name=\"vol1\">"));
        // Plane half-space boxes share one cube definition per sense.
        assert!(xml.contains("<box name=\"sl2p\""));
        assert!(xml.contains("<box name=\"sl3m\""));
        // The box cell is an intersection chain registered under csol2*.
        assert!(xml.contains("<intersection name=\"csol2"));
        assert!(xml.contains("<firstposition"));
        // World volume holds the universe-0 placements.
        assert!(xml.contains("<volume name=\"world\">"));
        assert!(xml.contains("<setup name=\"default\" version=\"1.0\">"));
        assert!(xml.contains("<world ref=\"world\"/>"));
        // Material stub and the half-space cutoff note.
        assert!(xml.contains("<material name=\"mat1\">"));
        assert!(drift
            .entries
            .iter()
            .any(|e| e.action == "halfspace-bounded" && e.scope == DriftScope::Deck));
        assert!(drift.entries.iter().any(|e| e.action == "material-stub"));
    }

    #[test]
    fn rpp_is_native_closed_box() {
        let (xml, drift) = translate(&deck_text(
            "1 1 -1.0 -1",
            "1 rpp -5 5 -5 5 -5 5",
            "m1 13027 1.0",
        ))
        .unwrap();
        assert!(xml.contains("<box name=\"sl1\" x=\"100\" y=\"100\" z=\"100\"/>"));
        assert!(!drift
            .entries
            .iter()
            .any(|e| e.action == "macrobody-expansion"));
    }

    #[test]
    fn one_cm_lengths_emit_as_ten_mm() {
        // Regression: cm values must convert x10 at emission — a 1 cm sphere
        // radius, a 1 cm RPP span, and a 1 cm translation all spell "10".
        let (xml, _) = translate(&deck_text(
            "1 1 -1.0 -1\n2 1 -1.0 -2",
            "1 so 1.0\n2 sph 0 2 0 0.5\n3 rpp 0 1 0 1 0 1",
            "m1 92235 1.0",
        ))
        .unwrap();
        assert!(xml.contains("<orb name=\"sl1\" r=\"10\"/>"), "{xml}");
        assert!(xml.contains("<orb name=\"sl2\" r=\"5\"/>"), "{xml}");
        assert!(
            xml.contains("<box name=\"sl3\" x=\"10\" y=\"10\" z=\"10\"/>"),
            "{xml}"
        );
        // The sphere centre (0, 2, 0) cm is a 20 mm y-offset position.
        assert!(
            xml.contains("x=\"0\" y=\"20\" z=\"0\" unit=\"mm\"/>"),
            "{xml}"
        );
    }

    #[test]
    fn rcc_emits_finite_tube_with_caps() {
        let (xml, drift) = translate(&deck_text(
            "1 1 -1.0 -1\n2 0 1",
            "1 rcc 0 0 -5 0 0 10 2",
            "m1 1001 1.0",
        ))
        .unwrap();
        // Interior: named boolean tube + caps; the tube is finite (z = |h|).
        // 2 cm radius / 10 cm height -> 20 mm / 100 mm.
        assert!(xml.contains(
            "<tube name=\"sl1t\" rmax=\"20\" deltaphi=\"6.283185307179586\" z=\"100\"/>"
        ));
        assert!(xml.contains("<intersection name=\"sl1\">"));
        assert!(drift
            .entries
            .iter()
            .any(|e| e.action == "macrobody-expansion"));
        // Exterior cell: subtraction from the cutoff box.
        assert!(xml.contains("<subtraction name=\"csol2o\">"));
    }

    #[test]
    fn canted_rcc_and_cones_are_loud() {
        let err = translate(&deck_text(
            "1 1 -1.0 -1",
            "1 rcc 0 0 0 1 1 1 2",
            "m1 1001 1.0",
        ))
        .unwrap_err();
        assert!(matches!(err, Error::MacrobodyOutOfScope { .. }), "{err}");
        let err =
            translate(&deck_text("1 1 -1.0 -1", "1 kz 0 0 0 1 1", "m1 1001 1.0")).unwrap_err();
        assert!(matches!(err, Error::UnsupportedSurface { .. }), "{err}");
    }

    #[test]
    fn box_axis_aligned_maps_and_canted_is_loud() {
        let (xml, _) = translate(&deck_text(
            "1 1 -1.0 -1",
            "1 box -5 -5 -5 10 0 0 0 10 0 0 0 10",
            "m1 1001 1.0",
        ))
        .unwrap();
        assert!(xml.contains("<box name=\"sl1\" x=\"100\" y=\"100\" z=\"100\"/>"));
        let err = translate(&deck_text(
            "1 1 -1.0 -1",
            "1 box 0 0 0 1 1 0 0 1 0 0 0 1",
            "m1 1001 1.0",
        ))
        .unwrap_err();
        assert!(matches!(err, Error::MacrobodyOutOfScope { .. }), "{err}");
    }

    #[test]
    fn cylinders_use_single_axis_firstrotations() {
        let (xml, _) = translate(&deck_text(
            "1 1 -1.0 -1 2 -3\n2 1 -1.0 1\n3 1 -1.0 2\n4 1 -1.0 3",
            "1 cx 2\n2 cy 3\n3 cz 4",
            "m1 1001 1.0",
        ))
        .unwrap();
        // CX rotates the z tube onto +x (and inversely on exterior cells);
        // CY onto +y; CZ stays unrotated.
        assert!(xml.contains("y=\"1.5707963267948966\""));
        assert!(xml.contains("y=\"-1.5707963267948966\""));
        assert!(xml.contains("x=\"1.5707963267948966\""));
        let n_rotations = xml.matches("<firstrotation").count();
        assert_eq!(n_rotations, 4, "two folds plus two exterior cells: {xml}");
    }

    #[test]
    fn complement_of_sphere_is_cutoff_minus_orb() {
        let (xml, drift) = translate(&deck_text(
            "1 1 -19.1 -1\n2 0 #1 -2",
            "1 so 10.0\n2 pz 0.0",
            "m1 92235 1.0",
        ))
        .unwrap();
        // #1 = outside the sphere, intersected with z<0: subtraction form.
        assert!(xml.contains("<subtraction name=\"csol2"));
        assert!(drift
            .entries
            .iter()
            .any(|e| e.action == "complement-expansion"));
    }

    #[test]
    fn union_of_exteriors_rewrites_through_cutoff_box() {
        // Cell 2 = outside sphere 1 OR outside sphere 2 (`:` union): both
        // exteriors, so the union must pass through the De Morgan rewrite.
        let (xml, _) = translate(&deck_text(
            "1 1 -1.0 -1\n2 0 1 : 2",
            "1 so 5\n2 so 7",
            "m1 1001 1.0",
        ))
        .unwrap();
        assert!(xml.contains("<subtraction name=\"dm2"), "{xml}");
        assert!(xml.contains("<intersection name=\"dm2"), "{xml}");
    }

    #[test]
    fn boundaries_are_loud() {
        let err = translate(&deck_text("1 0 -1", "*1 so 10", "")).unwrap_err();
        assert!(matches!(err, Error::GdmlBoundaryOutOfScope { .. }), "{err}");
        let err = translate(&deck_text("1 0 *-1", "1 so 10", "")).unwrap_err();
        assert!(matches!(err, Error::GdmlBoundaryOutOfScope { .. }), "{err}");
        let err = translate(&deck_text("1 0 -1 2", "1 -2 pz 0\n2 pz 5", "")).unwrap_err();
        assert!(matches!(err, Error::GdmlBoundaryOutOfScope { .. }), "{err}");
    }

    #[test]
    fn universes_become_assemblies() {
        let (xml, drift) = translate(&deck_text(
            "1 1 -1.0 -1 u=1\n2 0 -2 fill=1",
            "1 so 5\n2 so 10",
            "m1 92235 1.0",
        ))
        .unwrap();
        assert!(xml.contains("<assembly name=\"asm1\">"));
        assert!(xml.contains("<volumeref ref=\"vol1\"/>"));
        assert!(xml.contains("<volumeref ref=\"asm1\"/>"));
        assert!(drift.entries.iter().any(|e| e.action == "fill-applied"));
        assert!(drift
            .entries
            .iter()
            .any(|e| e.action == "universe-assigned"));
    }

    #[test]
    fn lattice_expands_to_element_placements() {
        let (xml, drift) = translate(&deck_text(
            "1 1 -1.0 -1 u=1\n2 1 -1.0 -2 u=2\n3 1 -1.0 -3 u=3\n4 1 -1.0 -4 u=4\n\
             10 0 -100 lat=1 fill=0:1 0:1 0:0 1 2 3 4\n20 0 #10",
            "1 sph 1 1 1 0.5\n2 sph 3 1 1 0.5\n3 sph 1 3 1 0.5\n4 sph 3 3 1 0.5\n\
             100 rpp 0 4 0 4 0 2",
            "m1 92235 1.0",
        ))
        .unwrap();
        // 2x2x1 = four element placements at the element centers (cm -> mm).
        assert_eq!(xml.matches("<volumeref ref=\"asm").count(), 4);
        assert!(xml.contains("x=\"10\" y=\"10\" z=\"10\" unit=\"mm\"/>"));
        assert!(xml.contains("x=\"30\" y=\"30\" z=\"10\" unit=\"mm\"/>"));
        assert!(drift
            .entries
            .iter()
            .any(|e| e.action == "lattice-expanded" && e.target == 10));
        assert!(drift
            .entries
            .iter()
            .any(|e| e.action == "lattice-emitted" && e.target == 10));
    }

    #[test]
    fn shared_loud_cases_match_other_directions() {
        // The lattice loud-case battery must reject identically.
        let element_cells =
            "1 1 -1.0 -1 u=1\n2 1 -1.0 -2 u=2\n3 1 -1.0 -3 u=3\n4 1 -1.0 -4 u=4\n20 0 #10";
        let element_surfs = "1 sph 1 1 1 0.5\n2 sph 3 1 1 0.5\n3 sph 1 3 1 0.5\n4 sph 3 3 1 0.5";
        for (lattice_cell, extra_surfs, data, is_transform, fragment) in [
            (
                "10 0 -100 lat=2 fill=0:1 0:1 0:0 1 2 3 4",
                "\n100 rpp 0 4 0 4 0 2",
                "m1 92235 1.0",
                false,
                "cell 10 parameter `lat=2`",
            ),
            (
                "10 0 -100 lat=1 fill=0:1 0:1 0:0 1 0 3 4",
                "\n100 rpp 0 4 0 4 0 2",
                "m1 92235 1.0",
                false,
                "cell 10 FILL matrix has a 0-hole",
            ),
            (
                "10 0 -100 lat=1 fill=0:1 0:1 0:0 1 2 3 4",
                "\n100 so 10",
                "m1 92235 1.0",
                false,
                "cell 10 lattice bounds",
            ),
        ] {
            let err = translate(&deck_text(
                &format!("{element_cells}\n{lattice_cell}"),
                &format!("{element_surfs}{extra_surfs}"),
                data,
            ))
            .unwrap_err();
            let text = err.to_string();
            assert!(
                if is_transform {
                    matches!(err, Error::TransformOutOfScope { .. })
                } else {
                    matches!(err, Error::UniverseOutOfScope { .. })
                } && text.contains(fragment),
                "{lattice_cell}: {err}"
            );
        }
    }
}
