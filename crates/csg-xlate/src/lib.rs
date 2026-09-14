#![warn(missing_docs)]
//! MCNP CSG to OpenMC `geometry.xml` translation (scoped v1).
//!
//! [`deck_csg_to_openmc_xml`] maps a parsed [`DeckProblem`] to an OpenMC
//! geometry document plus a [`DriftTable`] drift report. Only surface/cell
//! CSG plus a material stub (`material="void"` or the MCNP material number)
//! is translated; everything else is a loud [`Error`], never a silent skip.
//!
//! # GO scope (v1 maps, everything else errors)
//!
//! Surfaces (34 [`SurfKind`](nucleide_mcnp_io::surf::SurfKind) variants parsed
//! by `mcnp-io`):
//!
//! - `PX`/`X` to `x-plane`, `PY`/`Y` to `y-plane`, `PZ`/`Z` to `z-plane`.
//! - `SO`/`S`/`SX`/`SY`/`SZ` to `sphere` (center derived from the card).
//! - `CX`/`CY`/`CZ` (single-coefficient, on-axis) to the matching OpenMC
//!   cylinder with a zero transverse center.
//! - `SPH` to `sphere` directly; `RPP` expands to six axis-plane half-spaces;
//!   `RCC` expands to a cylinder plus two cap planes only when its axis is
//!   aligned with `x`, `y`, or `z`, else [`Error::MacrobodyOutOfScope`].
//! - `P` (three-point plane), `KX`/`KY`/`KZ` (five-coefficient cones with a
//!   sheet selector vs OpenMC's four-parameter cones), `SQ` (ten-coefficient
//!   ordering vs OpenMC `quadric` unverified), `GQ` (sixteen to ten
//!   reduction unverified), `TX`/`TY`/`TZ` (six-coefficient torus ordering
//!   unverified) to [`Error::UnsupportedSurface`].
//! - `BOX`/`REC`/`WED`/`RHP`/`HEX`/`TRC`/`ELL`/`ARB` and non-axis-aligned
//!   `RCC` to [`Error::MacrobodyOutOfScope`].
//!
//! Cells: juxtaposition intersection renders as whitespace, `:` union as
//! `|`, parentheses preserved. `#n` inlines the complement of cell `n`'s
//! region, but only when that region is a single half-space or an
//! intersection of half-spaces: De Morgan negation flips each half-space, so
//! no `~` operator appears in the output (macrobody expansions included).
//! Anything else is [`Error::ComplementTooComplex`]. Surface-card `*` sets
//! `boundary="reflective"`; cell-level `*n` strips the marker and forces the
//! surface reflective ([`Error::ReflectiveConflict`] on a periodic clash). A
//! surface periodic pointer sets `boundary="periodic"` plus
//! `periodic_surface_id` only when the partner exists, is not reflective,
//! and points back or nowhere, else [`Error::PeriodicAmbiguous`].
//! Transforms (`TRn`, surface pointers, `TRCL`), lattices, matrix or
//! transformed `FILL`s, `U=-n` no-truncate flags, tallies/sources, and
//! `read` includes are [`Error::TransformOutOfScope`],
//! [`Error::UniverseOutOfScope`], or [`Error::TallySourceOutOfScope`].
//! Reflecting and periodic boundaries have no verified Serpent mapping
//! ([`Error::SerpentBoundaryOutOfScope`]); the OpenMC direction keeps its
//! v1 boundary handling.
//!
//! # GO scope (v2 adds nested universes)
//!
//! Simple nested universes translate: cell `U=k` assignments (including
//! data-block `U` cards) set OpenMC `universe="k"`, and a transform-free
//! single-universe `FILL n` (cell param or data-block `FILL` card) sets
//! `fill="n"` with `material` omitted, per the OpenMC rule that a filled
//! cell carries no material. Drift notes `universe-assigned` and
//! `fill-applied` record each mapping. Everything else universe-shaped
//! stays loud: `LAT` lattices, matrix `FILL`s, `FILL` transforms,
//! `U=-n`, and `TRCL` on filled cells.
//!
//! # Example
//!
//! ```rust
//! use nucleide_csg_xlate::deck_csg_to_openmc_xml;
//! use nucleide_mcnp_io::problem::parse_deck;
//!
//! let deck = parse_deck(
//!     "msg\ntitle\n1 1 -1.0 -1\n\n1 so 10.0\n\nm1 92235 1.0\n",
//! )
//! .unwrap();
//! let (xml, drift) = deck_csg_to_openmc_xml(&deck).unwrap();
//! assert!(xml.contains("<geometry>"));
//! assert!(xml.contains("material=\"1\""));
//! assert!(drift.entries.is_empty());
//! ```

use std::collections::{BTreeMap, BTreeSet};

use nucleide_mcnp_io::cell::{CellCard, GeomExpr};
use nucleide_mcnp_io::problem::DeckProblem;
use nucleide_mcnp_io::semantic::split_card_name;
use nucleide_mcnp_io::surf::{SurfCard, SurfKind};

/// Crate-local result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors raised while translating MCNP CSG to OpenMC XML.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A surface kind has no verified v1 mapping (cones, quadrics, tori,
    /// general three-point planes).
    #[error("surface {surf} ({kind}) has no v1 mapping: {detail}")]
    UnsupportedSurface {
        /// MCNP surface number.
        surf: u32,
        /// Canonical surface-kind keyword.
        kind: String,
        /// Why v1 refuses it.
        detail: String,
    },
    /// A macrobody cannot be expanded to half-spaces in v1.
    #[error("surface {surf} ({kind}) is out of v1 scope: {detail}")]
    MacrobodyOutOfScope {
        /// MCNP surface number.
        surf: u32,
        /// Canonical surface-kind keyword.
        kind: String,
        /// Why v1 refuses it.
        detail: String,
    },
    /// A `#n` complement names a cell whose region is not a single
    /// half-space or an intersection of half-spaces.
    #[error("cell {cell} complement #{target} is too complex for v1 (unions and nested complements cannot inline)")]
    ComplementTooComplex {
        /// Cell carrying the `#n` token.
        cell: u32,
        /// Referenced cell number.
        target: i32,
    },
    /// A cell references a surface number that is neither a deck surface
    /// nor a v1-generated expansion facet.
    #[error("cell {cell} references unknown surface {surf}")]
    UnknownSurface {
        /// Cell carrying the reference.
        cell: u32,
        /// Missing surface number.
        surf: u32,
    },
    /// A `#n` complement names a cell number absent from the deck.
    #[error("cell {cell} complements unknown cell {target}")]
    UnknownCell {
        /// Cell carrying the `#n` token.
        cell: u32,
        /// Missing cell number.
        target: i32,
    },
    /// A surface is both reflective and periodic.
    #[error("surface {surf} is both reflective and periodic: {detail}")]
    ReflectiveConflict {
        /// Surface number.
        surf: u32,
        /// Which markers clashed.
        detail: String,
    },
    /// A periodic pointer cannot be paired unambiguously.
    #[error("surface {surf} periodic partner {partner} is ambiguous: {detail}")]
    PeriodicAmbiguous {
        /// Surface carrying the pointer.
        surf: u32,
        /// Named partner surface.
        partner: u32,
        /// Why pairing failed.
        detail: String,
    },
    /// A transform (`TRn` card, surface pointer, `TRCL`) needs translation.
    #[error("transforms are out of v1 scope: {detail}")]
    TransformOutOfScope {
        /// What carried the transform.
        detail: String,
    },
    /// Universes, lattices, `FILL`, or `read` includes need translation.
    ///
    /// Simple nested universes (`U=k`, single-universe `FILL n`) translate;
    /// lattices, matrix or transformed fills, and `U=-n` raise this error.
    #[error("universes/lattices/fills are out of v1 scope: {detail}")]
    UniverseOutOfScope {
        /// What carried the universe construct.
        detail: String,
    },
    /// A reflecting or periodic boundary has no verified Serpent mapping.
    #[error("surface {surf} boundary has no verified serpent mapping: {detail}")]
    SerpentBoundaryOutOfScope {
        /// Surface carrying the boundary marker.
        surf: u32,
        /// Why the Serpent direction refuses it.
        detail: String,
    },
    /// A periodic pointer has no PHITS spelling (reflective maps natively).
    #[error("surface {surf} periodic pointer has no phits spelling: {detail}")]
    PhitsBoundaryOutOfScope {
        /// Surface carrying the periodic pointer.
        surf: u32,
        /// Why the PHITS direction refuses it.
        detail: String,
    },
    /// A tally or source card needs translation.
    #[error("tallies/sources are out of v1 scope: {detail}")]
    TallySourceOutOfScope {
        /// Offending card name.
        detail: String,
    },
    /// MCNP deck validation failure (duplicates, dangling references).
    #[error(transparent)]
    Mcnp(#[from] nucleide_mcnp_io::inp::Error),
    /// XML serialization failure.
    #[error("XML write failed: {detail}")]
    Xml {
        /// Writer error text.
        detail: String,
    },
}

/// What a [`DriftEntry`] reports on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftScope {
    /// One cell (`target` is the cell number).
    Cell,
    /// One surface (`target` is the surface number).
    Surface,
    /// The deck as a whole (`target` is 0).
    Deck,
}

impl std::fmt::Display for DriftScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DriftScope::Cell => write!(f, "cell"),
            DriftScope::Surface => write!(f, "surface"),
            DriftScope::Deck => write!(f, "deck"),
        }
    }
}

/// One non-lossless but accepted translation step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriftEntry {
    /// Which object the entry reports on.
    pub scope: DriftScope,
    /// Cell/surface number (`0` for deck scope).
    pub target: u32,
    /// Machine-readable action (`"macrobody-expansion"`,
    /// `"complement-expansion"`, `"reflective-applied"`, `"periodic-link"`,
    /// `"universe-assigned"`, `"fill-applied"`, `"universe-data-card"`,
    /// `"dropped-cell-param"`, `"dropped-data-card"`).
    pub action: String,
    /// Human-readable reason.
    pub reason: String,
}

/// Per-cell/per-surface drift report for one translation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DriftTable {
    /// Drift entries in translation order (surfaces, then cells, then deck).
    pub entries: Vec<DriftEntry>,
}

impl DriftTable {
    /// Number of drift entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when the translation was fully lossless.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Entries for one scope.
    pub fn for_scope(&self, scope: DriftScope) -> Vec<&DriftEntry> {
        self.entries.iter().filter(|e| e.scope == scope).collect()
    }
}

/// OpenMC boundary condition carried by one translated surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Boundary {
    /// `boundary="reflective"`.
    Reflective,
    /// `boundary="periodic"` (paired in a second pass).
    Periodic,
}

/// One translated surface ready for XML emission.
struct OutSurface {
    /// OpenMC surface id (deck number or v1-allocated facet id).
    id: u32,
    /// OpenMC surface type (`x-plane`, `sphere`, ...).
    stype: &'static str,
    /// Coefficients in OpenMC order.
    coeffs: Vec<f64>,
    /// Boundary condition (`None` = transmission default, omitted).
    boundary: Option<Boundary>,
    /// Periodic partner id (set with [`Boundary::Periodic`]).
    periodic_id: Option<u32>,
}

/// How one deck surface maps to OpenMC.
enum SurfaceMap {
    /// Native surface with OpenMC-ordered coefficients.
    Direct {
        /// OpenMC surface type.
        stype: &'static str,
        /// Coefficients in OpenMC order.
        coeffs: Vec<f64>,
    },
    /// `RPP xmin xmax ymin ymax zmin zmax`.
    Rpp {
        /// Bounds in card order.
        bounds: [f64; 6],
    },
    /// `RCC base(3) axis(3) radius`.
    Rcc {
        /// Base point.
        base: [f64; 3],
        /// Axis vector (base to top).
        axis: [f64; 3],
        /// Radius.
        radius: f64,
    },
}

/// Map one deck surface to OpenMC, or refuse it loudly.
fn map_surface(card: &SurfCard) -> Result<SurfaceMap> {
    let c = &card.coeffs;
    match card.kind {
        SurfKind::Px | SurfKind::X => Ok(SurfaceMap::Direct {
            stype: "x-plane",
            coeffs: vec![c[0]],
        }),
        SurfKind::Py | SurfKind::Y => Ok(SurfaceMap::Direct {
            stype: "y-plane",
            coeffs: vec![c[0]],
        }),
        SurfKind::Pz | SurfKind::Z => Ok(SurfaceMap::Direct {
            stype: "z-plane",
            coeffs: vec![c[0]],
        }),
        SurfKind::So => Ok(SurfaceMap::Direct {
            stype: "sphere",
            coeffs: vec![0.0, 0.0, 0.0, c[0]],
        }),
        SurfKind::Sx => Ok(SurfaceMap::Direct {
            stype: "sphere",
            coeffs: vec![c[0], 0.0, 0.0, c[1]],
        }),
        SurfKind::Sy => Ok(SurfaceMap::Direct {
            stype: "sphere",
            coeffs: vec![0.0, c[0], 0.0, c[1]],
        }),
        SurfKind::Sz => Ok(SurfaceMap::Direct {
            stype: "sphere",
            coeffs: vec![0.0, 0.0, c[0], c[1]],
        }),
        SurfKind::S => Ok(SurfaceMap::Direct {
            stype: "sphere",
            coeffs: vec![c[0], c[1], c[2], c[3]],
        }),
        SurfKind::Cx => Ok(SurfaceMap::Direct {
            stype: "x-cylinder",
            coeffs: vec![0.0, 0.0, c[0]],
        }),
        SurfKind::Cy => Ok(SurfaceMap::Direct {
            stype: "y-cylinder",
            coeffs: vec![0.0, 0.0, c[0]],
        }),
        SurfKind::Cz => Ok(SurfaceMap::Direct {
            stype: "z-cylinder",
            coeffs: vec![0.0, 0.0, c[0]],
        }),
        SurfKind::Rpp => Ok(SurfaceMap::Rpp {
            bounds: [c[0], c[1], c[2], c[3], c[4], c[5]],
        }),
        SurfKind::Sph => Ok(SurfaceMap::Direct {
            stype: "sphere",
            coeffs: vec![c[0], c[1], c[2], c[3]],
        }),
        SurfKind::Rcc => Ok(SurfaceMap::Rcc {
            base: [c[0], c[1], c[2]],
            axis: [c[3], c[4], c[5]],
            radius: c[6],
        }),
        SurfKind::P => Err(Error::UnsupportedSurface {
            surf: card.num,
            kind: card.kind.keyword().to_string(),
            detail: "the 9-coefficient three-point plane has no verified v1 \
                derivation to OpenMC (A B C D)"
                .to_string(),
        }),
        SurfKind::Kx | SurfKind::Ky | SurfKind::Kz => Err(Error::UnsupportedSurface {
            surf: card.num,
            kind: card.kind.keyword().to_string(),
            detail: "the 5-coefficient cone (apex, slope, sheet selector) has \
                no lossless map to OpenMC's 4-parameter cone"
                .to_string(),
        }),
        SurfKind::Sq => Err(Error::UnsupportedSurface {
            surf: card.num,
            kind: card.kind.keyword().to_string(),
            detail: "the 10-coefficient special-quadric ordering against \
                OpenMC quadric is unverified"
                .to_string(),
        }),
        SurfKind::Gq => Err(Error::UnsupportedSurface {
            surf: card.num,
            kind: card.kind.keyword().to_string(),
            detail: "the 16-coefficient general-quadric reduction to OpenMC's \
                10-parameter quadric is unverified"
                .to_string(),
        }),
        SurfKind::Tx | SurfKind::Ty | SurfKind::Tz => Err(Error::UnsupportedSurface {
            surf: card.num,
            kind: card.kind.keyword().to_string(),
            detail: "the 6-coefficient torus ordering against OpenMC torus \
                (x0 y0 z0 A B C) is unverified"
                .to_string(),
        }),
        SurfKind::McBox
        | SurfKind::Rec
        | SurfKind::Wed
        | SurfKind::Rhp
        | SurfKind::Hex
        | SurfKind::Trc
        | SurfKind::Ell
        | SurfKind::Arb => Err(Error::MacrobodyOutOfScope {
            surf: card.num,
            kind: card.kind.keyword().to_string(),
            detail: "v1 expands RPP/SPH/axis-aligned RCC only".to_string(),
        }),
    }
}

/// Resolved OpenMC region: macrobody and complement references expanded.
enum Resolved {
    /// Signed surface half-space (positive renders bare, negative `-id`).
    Lit(i32),
    /// Whitespace intersection.
    And(Vec<Resolved>),
    /// `|` union.
    Or(Box<Resolved>, Box<Resolved>),
}

impl Resolved {
    /// Render with minimal parentheses (unions parenthesized in intersections).
    fn render(&self) -> String {
        match self {
            Resolved::Lit(s) => {
                if *s < 0 {
                    format!("-{}", s.unsigned_abs())
                } else {
                    format!("{s}")
                }
            }
            Resolved::And(parts) => parts
                .iter()
                .map(|p| match p {
                    Resolved::Or(..) => format!("({})", p.render()),
                    _ => p.render(),
                })
                .collect::<Vec<_>>()
                .join(" "),
            Resolved::Or(a, b) => format!("{} | {}", a.render(), b.render()),
        }
    }
}

/// Translation working state shared by surface and cell passes.
struct Ctx<'a> {
    /// Deck cells by number.
    cells: BTreeMap<u32, &'a CellCard>,
    /// Deck surfaces by number.
    surfs: BTreeMap<u32, &'a SurfCard>,
    /// Macrobody surface number to expansion facet data.
    macros: BTreeMap<u32, MacroExpansion>,
    /// Next synthetic surface id (above every deck surface number).
    next_id: u32,
    /// Surfaces forced reflective by cell-level `*n` markers.
    forced_reflective: BTreeSet<u32>,
    /// Drift entries.
    drift: DriftTable,
}

/// Half-space expansion of one macrobody surface.
struct MacroExpansion {
    /// Facet surface ids with the sense selecting the macrobody interior.
    facets: Vec<OutSurface>,
    /// Signed facet ids whose intersection is the interior (`-id`/`id`).
    interior: Vec<i32>,
}

impl<'a> Ctx<'a> {
    /// Push a drift entry.
    fn note(&mut self, scope: DriftScope, target: u32, action: &str, reason: String) {
        self.drift.entries.push(DriftEntry {
            scope,
            target,
            action: action.to_string(),
            reason,
        });
    }
}

/// Expand an `RPP` surface to six axis planes.
fn expand_rpp(
    next_id: &mut u32,
    drift: &mut DriftTable,
    card: &SurfCard,
    bounds: [f64; 6],
) -> MacroExpansion {
    let kinds = [
        ("x-plane", bounds[0]),
        ("x-plane", bounds[1]),
        ("y-plane", bounds[2]),
        ("y-plane", bounds[3]),
        ("z-plane", bounds[4]),
        ("z-plane", bounds[5]),
    ];
    let mut facets = Vec::with_capacity(6);
    let mut interior = Vec::with_capacity(6);
    for (i, (stype, coeff)) in kinds.iter().enumerate() {
        let id = *next_id;
        *next_id += 1;
        facets.push(OutSurface {
            id,
            stype,
            coeffs: vec![*coeff],
            boundary: None,
            periodic_id: None,
        });
        // Even index = minimum bound (positive sense inside), odd = maximum.
        interior.push(if i % 2 == 0 { id as i32 } else { -(id as i32) });
    }
    drift.entries.push(DriftEntry {
        scope: DriftScope::Surface,
        target: card.num,
        action: "macrobody-expansion".to_string(),
        reason: format!(
            "RPP {} expanded to 6 axis planes {}",
            card.num,
            facets
                .iter()
                .map(|f| f.id.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        ),
    });
    MacroExpansion { facets, interior }
}

/// Expand an axis-aligned `RCC` to a cylinder plus two cap planes.
fn expand_rcc(
    next_id: &mut u32,
    drift: &mut DriftTable,
    card: &SurfCard,
    base: [f64; 3],
    axis: [f64; 3],
    radius: f64,
) -> Result<MacroExpansion> {
    let nonzero: Vec<usize> = (0..3).filter(|&i| axis[i] != 0.0).collect();
    if nonzero.len() != 1 {
        return Err(Error::MacrobodyOutOfScope {
            surf: card.num,
            kind: card.kind.keyword().to_string(),
            detail: "only RCCs aligned with x, y, or z expand in v1 \
                (canted cylinders have no verified OpenMC spelling)"
                .to_string(),
        });
    }
    let ax = nonzero[0];
    let (cyl_type, cyl_coeffs, lo, hi) = match ax {
        0 => (
            "x-cylinder",
            vec![base[1], base[2], radius],
            base[0],
            base[0] + axis[0],
        ),
        1 => (
            "y-cylinder",
            vec![base[0], base[2], radius],
            base[1],
            base[1] + axis[1],
        ),
        _ => (
            "z-cylinder",
            vec![base[0], base[1], radius],
            base[2],
            base[2] + axis[2],
        ),
    };
    let plane_type = ["x-plane", "y-plane", "z-plane"][ax];
    let cyl_id = *next_id;
    *next_id += 1;
    let lo_id = *next_id;
    *next_id += 1;
    let hi_id = *next_id;
    *next_id += 1;
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    let facets = vec![
        OutSurface {
            id: cyl_id,
            stype: cyl_type,
            coeffs: cyl_coeffs,
            boundary: None,
            periodic_id: None,
        },
        OutSurface {
            id: lo_id,
            stype: plane_type,
            coeffs: vec![lo],
            boundary: None,
            periodic_id: None,
        },
        OutSurface {
            id: hi_id,
            stype: plane_type,
            coeffs: vec![hi],
            boundary: None,
            periodic_id: None,
        },
    ];
    drift.entries.push(DriftEntry {
        scope: DriftScope::Surface,
        target: card.num,
        action: "macrobody-expansion".to_string(),
        reason: format!(
            "RCC {} expanded to {} {} and 2 cap planes ({} {})",
            card.num, cyl_type, cyl_id, lo_id, hi_id
        ),
    });
    Ok(MacroExpansion {
        facets,
        interior: vec![-(cyl_id as i32), lo_id as i32, -(hi_id as i32)],
    })
}

/// True when an expression is a single half-space or an intersection of
/// half-spaces (the only complement targets v1 inlines).
fn is_flat(expr: &GeomExpr) -> bool {
    match expr {
        GeomExpr::HalfSpace(_) => true,
        GeomExpr::Intersect(parts) => parts.iter().all(is_flat),
        GeomExpr::Union(..) | GeomExpr::Complement(..) => false,
    }
}

/// Resolve one half-space literal: strip `*`, expand macrobodies.
fn resolve_lit(ctx: &mut Ctx<'_>, cell_num: u32, surf: i32, reflecting: bool) -> Result<Resolved> {
    let id = surf.unsigned_abs();
    if reflecting {
        ctx.forced_reflective.insert(id);
        ctx.note(
            DriftScope::Cell,
            cell_num,
            "reflective-applied",
            format!("cell-level *{id} stripped; surface {id} forced reflective"),
        );
    }
    if ctx.surfs.contains_key(&id) && !ctx.macros.contains_key(&id) {
        return Ok(Resolved::Lit(surf));
    }
    if let Some(expansion) = ctx.macros.get(&id) {
        let interior = expansion.interior.clone();
        if reflecting {
            for facet in &interior {
                ctx.forced_reflective.insert(facet.unsigned_abs());
            }
        }
        let lits: Vec<Resolved> = interior
            .into_iter()
            .map(|s| Resolved::Lit(if surf < 0 { s } else { -s }))
            .collect();
        if surf < 0 {
            return Ok(Resolved::And(lits));
        }
        // Positive sense of a macrobody is the union of outside half-spaces.
        let mut iter = lits.into_iter();
        let first = iter.next().expect("macrobody expansion is non-empty");
        return Ok(iter.fold(first, |a, b| Resolved::Or(Box::new(a), Box::new(b))));
    }
    Err(Error::UnknownSurface {
        cell: cell_num,
        surf: id,
    })
}

/// Resolve a `#n` complement by inlining the referenced region (De Morgan).
fn resolve_complement(ctx: &mut Ctx<'_>, cell_num: u32, target: i32) -> Result<Resolved> {
    if target <= 0 {
        return Err(Error::UnknownCell {
            cell: cell_num,
            target,
        });
    }
    let target_num = target as u32;
    let referenced = ctx
        .cells
        .get(&target_num)
        .copied()
        .ok_or(Error::UnknownCell {
            cell: cell_num,
            target,
        })?;
    if !is_flat(&referenced.geom) {
        return Err(Error::ComplementTooComplex {
            cell: cell_num,
            target,
        });
    }
    ctx.note(
        DriftScope::Cell,
        cell_num,
        "complement-expansion",
        format!(
            "#{} inlined as the negated region of cell {}",
            target, target
        ),
    );
    // Collect the referenced half-spaces, then negate each one.
    fn lits(expr: &GeomExpr, out: &mut Vec<(i32, bool)>) {
        match expr {
            GeomExpr::HalfSpace(h) => out.push((h.surf, h.reflecting)),
            GeomExpr::Intersect(parts) => parts.iter().for_each(|p| lits(p, out)),
            GeomExpr::Union(..) | GeomExpr::Complement(..) => {
                unreachable!("flat regions hold no unions or complements")
            }
        }
    }
    let mut raw = Vec::new();
    lits(&referenced.geom, &mut raw);
    let mut negated = Vec::with_capacity(raw.len());
    for (surf, reflecting) in raw {
        // Negating the sense before resolving turns the De Morgan dual into
        // a direct resolution: the negation of one half-space is the flipped
        // half-space (no `~` needed), `-(+M)` is the interior intersection,
        // and `-(-M)` the exterior union.
        negated.push(resolve_lit(ctx, cell_num, -surf, reflecting)?);
    }
    let mut iter = negated.into_iter();
    let first = iter.next().expect("flat regions hold at least one literal");
    Ok(iter.fold(first, |a, b| Resolved::Or(Box::new(a), Box::new(b))))
}

/// Resolve one cell geometry expression to OpenMC region form.
fn resolve_expr(ctx: &mut Ctx<'_>, cell_num: u32, expr: &GeomExpr) -> Result<Resolved> {
    match expr {
        GeomExpr::HalfSpace(h) => resolve_lit(ctx, cell_num, h.surf, h.reflecting),
        GeomExpr::Intersect(parts) => {
            let mut out = Vec::with_capacity(parts.len());
            for part in parts {
                out.push(resolve_expr(ctx, cell_num, part)?);
            }
            Ok(Resolved::And(out))
        }
        GeomExpr::Union(a, b) => Ok(Resolved::Or(
            Box::new(resolve_expr(ctx, cell_num, a)?),
            Box::new(resolve_expr(ctx, cell_num, b)?),
        )),
        GeomExpr::Complement(inner) => match inner.as_ref() {
            GeomExpr::HalfSpace(h) => resolve_complement(ctx, cell_num, h.surf),
            _ => Err(Error::ComplementTooComplex {
                cell: cell_num,
                target: -1,
            }),
        },
    }
}

/// One translated cell ready for XML emission.
struct OutCell {
    /// MCNP cell number.
    id: u32,
    /// MCNP material number (`0` = void).
    mat: u32,
    /// Containing universe (`0` when the cell carries no `U=k`).
    universe: u32,
    /// Filling universe for a single-universe `FILL n` (`None` otherwise).
    fill: Option<u32>,
    /// Rendered OpenMC region string.
    region: String,
}

/// Reject cell-level lattice/transform parameters and note dropped ones.
///
/// `U=k` and single-universe `FILL n` are honored later via the semantic
/// universe/fill views (see [`resolve_universes`]); this pass only notes
/// them in drift and rejects what v2 cannot carry.
fn check_cell_params(ctx: &mut Ctx<'_>, cell: &CellCard) -> Result<()> {
    for param in &cell.params {
        let Some((key, _)) = param.split_once('=') else {
            // Bare keywords (`vol`, `pwt`, ...) carry no geometry.
            ctx.note(
                DriftScope::Cell,
                cell.num,
                "dropped-cell-param",
                format!("bare cell parameter `{param}` not represented in OpenMC geometry"),
            );
            continue;
        };
        let base = key
            .trim_start_matches('*')
            .split(':')
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        match base.as_str() {
            "u" | "fill" => {
                // Honored via resolve_universes; matrix/transform shapes are
                // rejected there with cell-qualified details.
                continue;
            }
            "lat" => {
                return Err(Error::UniverseOutOfScope {
                    detail: format!(
                        "cell {} parameter `{param}` needs lattice translation",
                        cell.num
                    ),
                });
            }
            "trcl" => {
                return Err(Error::TransformOutOfScope {
                    detail: format!(
                        "cell {} parameter `{param}` needs transform translation",
                        cell.num
                    ),
                });
            }
            _ => {
                ctx.note(
                    DriftScope::Cell,
                    cell.num,
                    "dropped-cell-param",
                    format!("cell parameter `{param}` not represented in OpenMC geometry"),
                );
            }
        }
    }
    Ok(())
}

/// Tally/source card prefixes that carry a card number.
const TALLY_PREFIXES: [&str; 9] = ["F", "FM", "FC", "E", "DE", "DF", "FT", "FU", "FQ"];

/// Reject out-of-scope data cards and note dropped ones.
fn check_data_cards(ctx: &mut Ctx<'_>, deck: &DeckProblem) -> Result<()> {
    for card in &deck.data {
        if card.name.is_empty() {
            continue; // Comment/blank passthrough.
        }
        let named = split_card_name(&card.name);
        let prefix = named.prefix.as_str();
        if prefix == "M" && named.number.is_some() {
            continue; // Material stub source.
        }
        if prefix == "TR" && named.number.is_some() {
            return Err(Error::TransformOutOfScope {
                detail: format!("{} card needs transform translation", card.name),
            });
        }
        if (TALLY_PREFIXES.contains(&prefix) && named.number.is_some())
            || matches!(prefix, "SDEF" | "KCODE" | "KSRC" | "SSR" | "SSW" | "FMESH")
        {
            return Err(Error::TallySourceOutOfScope {
                detail: format!("{} card needs tally/source translation", card.name),
            });
        }
        if matches!(prefix, "U" | "FILL") && named.classifier.is_empty() {
            // Honored via resolve_universes; noted once per card.
            ctx.note(
                DriftScope::Deck,
                0,
                "universe-data-card",
                format!("{} card honored as universe assignment", card.name),
            );
            continue;
        }
        if matches!(prefix, "LAT") && named.classifier.is_empty() {
            return Err(Error::UniverseOutOfScope {
                detail: format!("{} card needs lattice translation", card.name),
            });
        }
        if card.name.eq_ignore_ascii_case("READ") {
            return Err(Error::UniverseOutOfScope {
                detail: "read includes are never followed, so the deck is incomplete".to_string(),
            });
        }
        ctx.note(
            DriftScope::Deck,
            0,
            "dropped-data-card",
            format!("{} card not represented in OpenMC geometry", card.name),
        );
    }
    Ok(())
}

/// Resolve per-cell universes and single-universe fills via the semantic views.
///
/// Returns `(cell_universe, cell_fill)` maps; cells absent from both live in
/// universe 0 with no fill. Lattices, matrix fills, fill transforms, and
/// `U=-n` no-truncate flags stay loud [`Error::UniverseOutOfScope`] (or
/// [`Error::TransformOutOfScope`] for transforms).
fn resolve_universes(
    ctx: &mut Ctx<'_>,
    deck: &DeckProblem,
) -> Result<(BTreeMap<u32, u32>, BTreeMap<u32, u32>)> {
    use nucleide_mcnp_io::semantic::FillTarget;
    if let Some(lattice) = deck.lattices()?.first() {
        return Err(Error::UniverseOutOfScope {
            detail: format!(
                "cell {} LAT={} needs lattice translation",
                lattice.cell, lattice.lattice
            ),
        });
    }
    let mut universes: BTreeMap<u32, u32> = BTreeMap::new();
    for view in deck.universes()? {
        if !view.not_truncated.is_empty() {
            return Err(Error::UniverseOutOfScope {
                detail: format!(
                    "cells {:?} use U=-{} with no OpenMC equivalent",
                    view.not_truncated, view.number
                ),
            });
        }
        for cell in &view.cells {
            universes.insert(*cell, view.number);
        }
    }
    let mut fills: BTreeMap<u32, u32> = BTreeMap::new();
    for view in deck.fills()? {
        if view.transform.is_some() {
            return Err(Error::TransformOutOfScope {
                detail: format!(
                    "cell {} FILL transform needs transform translation",
                    view.cell
                ),
            });
        }
        match view.target {
            FillTarget::Single(universe) => {
                fills.insert(view.cell, universe);
            }
            FillTarget::Matrix { .. } => {
                return Err(Error::UniverseOutOfScope {
                    detail: format!("cell {} FILL matrix needs lattice translation", view.cell),
                });
            }
        }
    }
    for (cell, universe) in &universes {
        if *universe != 0 {
            ctx.note(
                DriftScope::Cell,
                *cell,
                "universe-assigned",
                format!("cell {cell} assigned to universe {universe}"),
            );
        }
    }
    for (cell, universe) in &fills {
        ctx.note(
            DriftScope::Cell,
            *cell,
            "fill-applied",
            format!("cell {cell} filled with universe {universe}"),
        );
    }
    Ok((universes, fills))
}

/// Translate a parsed MCNP deck to OpenMC `geometry.xml` plus drift.
///
/// Surfaces, cells, simple nested universes, and a material stub
/// (`material="void"` for `mat == 0`, else the MCNP material number)
/// translate; a filled cell carries `fill` instead of `material`.
/// Transforms, lattices, matrix or transformed fills, tallies, and sources
/// are loud [`Error`]s. The deck is
/// validated first, so duplicate numbers and dangling surface, complement,
/// material, periodic, and transform links fail as [`Error::Mcnp`].
pub fn deck_csg_to_openmc_xml(deck: &DeckProblem) -> Result<(String, DriftTable)> {
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
    let (cell_universe, cell_fill) = resolve_universes(&mut ctx, deck)?;

    // Surface pass: reject transforms, map kinds, expand macrobodies.
    let mut periodic: BTreeMap<u32, u32> = BTreeMap::new();
    let mut card_reflective: BTreeSet<u32> = BTreeSet::new();
    let mut direct: Vec<OutSurface> = Vec::with_capacity(deck.surfs.len());
    for card in &deck.surfs {
        if let Some(tr) = card.transform {
            return Err(Error::TransformOutOfScope {
                detail: format!("surface {} links transform {tr}", card.num),
            });
        }
        if let Some(partner) = card.periodic {
            periodic.insert(card.num, partner);
        }
        if card.reflecting {
            card_reflective.insert(card.num);
        }
        match map_surface(card)? {
            SurfaceMap::Direct { stype, coeffs } => direct.push(OutSurface {
                id: card.num,
                stype,
                coeffs,
                boundary: None,
                periodic_id: None,
            }),
            SurfaceMap::Rpp { bounds } => {
                let expansion = expand_rpp(&mut ctx.next_id, &mut ctx.drift, card, bounds);
                ctx.macros.insert(card.num, expansion);
            }
            SurfaceMap::Rcc { base, axis, radius } => {
                let expansion =
                    expand_rcc(&mut ctx.next_id, &mut ctx.drift, card, base, axis, radius)?;
                ctx.macros.insert(card.num, expansion);
            }
        }
    }

    // Cell pass: resolve regions (collects forced-reflective markers).
    let mut regions: Vec<OutCell> = Vec::with_capacity(deck.cells.len());
    for cell in &deck.cells {
        let resolved = resolve_expr(&mut ctx, cell.num, &cell.geom)?;
        regions.push(OutCell {
            id: cell.num,
            mat: cell.mat,
            universe: cell_universe.get(&cell.num).copied().unwrap_or(0),
            fill: cell_fill.get(&cell.num).copied(),
            region: resolved.render(),
        });
    }

    // Boundary pass: reflective markers, then periodic pairing.
    let mut surfaces: Vec<OutSurface> = direct;
    for expansion in ctx.macros.values() {
        surfaces.extend(expansion.facets.iter().map(|f| OutSurface {
            id: f.id,
            stype: f.stype,
            coeffs: f.coeffs.clone(),
            boundary: None,
            periodic_id: None,
        }));
    }
    let known: BTreeSet<u32> = surfaces.iter().map(|s| s.id).collect();
    for id in &ctx.forced_reflective {
        if !known.contains(id) {
            return Err(Error::UnknownSurface { cell: 0, surf: *id });
        }
    }
    for (surf, partner) in &periodic {
        if !ctx.surfs.contains_key(partner) {
            return Err(Error::PeriodicAmbiguous {
                surf: *surf,
                partner: *partner,
                detail: format!("partner surface {partner} is absent from the deck"),
            });
        }
        if card_reflective.contains(surf) || ctx.forced_reflective.contains(surf) {
            return Err(Error::ReflectiveConflict {
                surf: *surf,
                detail: format!("surface {surf} is both reflective and periodic"),
            });
        }
        if card_reflective.contains(partner) || ctx.forced_reflective.contains(partner) {
            return Err(Error::ReflectiveConflict {
                surf: *partner,
                detail: format!("periodic partner {partner} is reflective"),
            });
        }
        match periodic.get(partner) {
            Some(back) if *back != *surf => {
                return Err(Error::PeriodicAmbiguous {
                    surf: *surf,
                    partner: *partner,
                    detail: format!("partner {partner} points at surface {back} instead"),
                });
            }
            _ => {}
        }
    }
    for out in &mut surfaces {
        if card_reflective.contains(&out.id) || ctx.forced_reflective.contains(&out.id) {
            out.boundary = Some(Boundary::Reflective);
        }
    }
    // Pair periodic links symmetrically (one-sided pointers pair implicitly).
    let mut linked: BTreeSet<(u32, u32)> = BTreeSet::new();
    for (surf, partner) in &periodic {
        for out in &mut surfaces {
            if &out.id == surf {
                out.boundary = Some(Boundary::Periodic);
                out.periodic_id = Some(*partner);
            } else if &out.id == partner {
                out.boundary = Some(Boundary::Periodic);
                out.periodic_id = Some(*surf);
            }
        }
        if linked.insert((*surf, *partner)) {
            ctx.note(
                DriftScope::Surface,
                *surf,
                "periodic-link",
                format!("surface {surf} paired periodic with surface {partner}"),
            );
        }
    }
    surfaces.sort_by_key(|s| s.id);

    let xml = emit_xml(&surfaces, &regions)?;
    Ok((xml, ctx.drift))
}

/// Render the OpenMC `geometry.xml` document with `quick-xml`.
fn emit_xml(surfaces: &[OutSurface], regions: &[OutCell]) -> Result<String> {
    use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, Event};
    use quick_xml::writer::Writer;

    fn xml_error(e: impl std::fmt::Display) -> Error {
        Error::Xml {
            detail: e.to_string(),
        }
    }
    let mut writer = Writer::new_with_indent(Vec::new(), b' ', 2);
    writer
        .write_event(Event::Decl(BytesDecl::new("1.0", Some("utf-8"), None)))
        .map_err(xml_error)?;
    writer
        .write_event(Event::Start(BytesStart::new("geometry")))
        .map_err(xml_error)?;
    for surf in surfaces {
        let mut elem = BytesStart::new("surface");
        let id = surf.id.to_string();
        elem.push_attribute(("id", id.as_str()));
        elem.push_attribute(("type", surf.stype));
        let coeffs = surf
            .coeffs
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" ");
        elem.push_attribute(("coeffs", coeffs.as_str()));
        let boundary = match surf.boundary {
            Some(Boundary::Reflective) => Some("reflective"),
            Some(Boundary::Periodic) => Some("periodic"),
            None => None,
        };
        if let Some(boundary) = boundary {
            elem.push_attribute(("boundary", boundary));
        }
        if let Some(partner) = surf.periodic_id {
            let text = partner.to_string();
            elem.push_attribute(("periodic_surface_id", text.as_str()));
        }
        writer.write_event(Event::Empty(elem)).map_err(xml_error)?;
    }
    for out in regions {
        let mut elem = BytesStart::new("cell");
        let id = out.id.to_string();
        elem.push_attribute(("id", id.as_str()));
        let universe = out.universe.to_string();
        elem.push_attribute(("universe", universe.as_str()));
        if let Some(fill) = out.fill {
            // OpenMC: a filled cell carries no material.
            let text = fill.to_string();
            elem.push_attribute(("fill", text.as_str()));
        } else {
            let material = if out.mat == 0 {
                "void".to_string()
            } else {
                out.mat.to_string()
            };
            elem.push_attribute(("material", material.as_str()));
        }
        elem.push_attribute(("region", out.region.as_str()));
        writer.write_event(Event::Empty(elem)).map_err(xml_error)?;
    }
    writer
        .write_event(Event::End(BytesEnd::new("geometry")))
        .map_err(xml_error)?;
    String::from_utf8(writer.into_inner()).map_err(xml_error)
}
/// Map one deck surface to a Serpent `surf` card body.
///
/// Returns the Serpent surface type plus parameters in Serpent order.
/// `RPP` maps to native `cuboid` and axis-aligned `RCC` to the truncated
/// `cylx`/`cyly`/`cylz` forms, so unlike the OpenMC direction nothing
/// expands. Out-of-scope kinds reuse [`Error::UnsupportedSurface`] and
/// [`Error::MacrobodyOutOfScope`] with Serpent-flavored details.
fn map_serpent_surface(card: &SurfCard) -> Result<(&'static str, Vec<f64>)> {
    let c = &card.coeffs;
    match card.kind {
        SurfKind::Px | SurfKind::X => Ok(("px", vec![c[0]])),
        SurfKind::Py | SurfKind::Y => Ok(("py", vec![c[0]])),
        SurfKind::Pz | SurfKind::Z => Ok(("pz", vec![c[0]])),
        SurfKind::So => Ok(("sph", vec![0.0, 0.0, 0.0, c[0]])),
        SurfKind::Sx => Ok(("sph", vec![c[0], 0.0, 0.0, c[1]])),
        SurfKind::Sy => Ok(("sph", vec![0.0, c[0], 0.0, c[1]])),
        SurfKind::Sz => Ok(("sph", vec![0.0, 0.0, c[0], c[1]])),
        SurfKind::S => Ok(("sph", vec![c[0], c[1], c[2], c[3]])),
        SurfKind::Cx => Ok(("cylx", vec![0.0, 0.0, c[0]])),
        SurfKind::Cy => Ok(("cyly", vec![0.0, 0.0, c[0]])),
        SurfKind::Cz => Ok(("cylz", vec![0.0, 0.0, c[0]])),
        SurfKind::Sph => Ok(("sph", vec![c[0], c[1], c[2], c[3]])),
        SurfKind::Rpp => Ok(("cuboid", vec![c[0], c[1], c[2], c[3], c[4], c[5]])),
        SurfKind::Rcc => {
            let base = [c[0], c[1], c[2]];
            let axis = [c[3], c[4], c[5]];
            let radius = c[6];
            let nonzero: Vec<usize> = axis
                .iter()
                .enumerate()
                .filter(|(_, v)| **v != 0.0)
                .map(|(i, _)| i)
                .collect();
            if nonzero.len() != 1 {
                return Err(Error::MacrobodyOutOfScope {
                    surf: card.num,
                    kind: card.kind.keyword().to_string(),
                    detail: "only axis-aligned RCC maps to Serpent truncated cylinders".to_string(),
                });
            }
            let a = nonzero[0];
            let lo = base[a];
            Ok(match a {
                0 => ("cylx", vec![base[1], base[2], radius, lo, lo + axis[0]]),
                1 => ("cyly", vec![base[0], base[2], radius, lo, lo + axis[1]]),
                _ => ("cylz", vec![base[0], base[1], radius, lo, lo + axis[2]]),
            })
        }
        SurfKind::P => Err(Error::UnsupportedSurface {
            surf: card.num,
            kind: card.kind.keyword().to_string(),
            detail: "the 9-coefficient three-point plane against Serpent \
                plane/mplane (right-hand-rule sidedness) is unverified"
                .to_string(),
        }),
        SurfKind::Kx | SurfKind::Ky | SurfKind::Kz => Err(Error::UnsupportedSurface {
            surf: card.num,
            kind: card.kind.keyword().to_string(),
            detail: "the 5-coefficient cone (apex, slope, sheet selector) \
                against Serpent cone/ckx/cky/ckz is unverified"
                .to_string(),
        }),
        SurfKind::Sq | SurfKind::Gq => Err(Error::UnsupportedSurface {
            surf: card.num,
            kind: card.kind.keyword().to_string(),
            detail: "the MCNP quadric coefficient ordering against Serpent \
                quadratic is unverified"
                .to_string(),
        }),
        SurfKind::Tx | SurfKind::Ty | SurfKind::Tz => Err(Error::UnsupportedSurface {
            surf: card.num,
            kind: card.kind.keyword().to_string(),
            detail: "the 6-coefficient torus ordering against Serpent \
                torx/tory/torz (elliptical) is unverified"
                .to_string(),
        }),
        SurfKind::McBox
        | SurfKind::Rec
        | SurfKind::Wed
        | SurfKind::Rhp
        | SurfKind::Hex
        | SurfKind::Trc
        | SurfKind::Ell
        | SurfKind::Arb => Err(Error::MacrobodyOutOfScope {
            surf: card.num,
            kind: card.kind.keyword().to_string(),
            detail: "serpent maps planes, spheres, cylinders, cuboid, and \
                axis-aligned RCC only"
                .to_string(),
        }),
    }
}

/// Reject reflecting markers and periodic pointers for the Serpent
/// direction, which has no verified per-surface boundary mapping.
fn reject_serpent_boundaries(deck: &DeckProblem) -> Result<()> {
    fn walk(cell: u32, expr: &GeomExpr) -> Result<()> {
        match expr {
            GeomExpr::HalfSpace(h) => {
                if h.reflecting {
                    return Err(Error::SerpentBoundaryOutOfScope {
                        surf: h.surf.unsigned_abs(),
                        detail: format!("cell {cell} `*` marker has no serpent spelling"),
                    });
                }
                Ok(())
            }
            GeomExpr::Intersect(parts) => {
                for part in parts {
                    walk(cell, part)?;
                }
                Ok(())
            }
            GeomExpr::Union(a, b) => {
                walk(cell, a)?;
                walk(cell, b)
            }
            // `#n` passes through natively; the target cell's own markers
            // are checked when that cell is walked.
            GeomExpr::Complement(_) => Ok(()),
        }
    }
    for card in &deck.surfs {
        if card.reflecting {
            return Err(Error::SerpentBoundaryOutOfScope {
                surf: card.num,
                detail: "reflective surface card has no serpent spelling".to_string(),
            });
        }
        if card.periodic.is_some() {
            return Err(Error::SerpentBoundaryOutOfScope {
                surf: card.num,
                detail: "periodic surface pointer has no serpent spelling".to_string(),
            });
        }
    }
    for cell in &deck.cells {
        walk(cell.num, &cell.geom)?;
    }
    Ok(())
}

/// Translate a parsed MCNP deck to Serpent input (`surf`/`cell` cards) plus drift.
///
/// Same v2 scope as the OpenMC direction (surfaces, cells, simple nested
/// universes, material-name stub), with three Serpent-native simplifications:
/// `RPP`/`RCC` need no expansion (`cuboid`, truncated cylinders), `#n`
/// passes through as Serpent's native cell complement, and the region text
/// is the MCNP boolean spelling Serpent shares (juxtaposition, `:`, parens).
/// Material `mat == 0` renders as `void`, else as `m<mat>` (caller supplies
/// the `mat` cards); a filled cell renders `fill <n>` with no material. An
/// empty region synthesizes an `inf` surface. Reflecting and periodic
/// boundaries are loud [`Error::SerpentBoundaryOutOfScope`]s.
pub fn deck_csg_to_serpent_input(deck: &DeckProblem) -> Result<(String, DriftTable)> {
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
    let (cell_universe, cell_fill) = resolve_universes(&mut ctx, deck)?;
    reject_serpent_boundaries(deck)?;
    for card in &deck.surfs {
        if let Some(tr) = card.transform {
            return Err(Error::TransformOutOfScope {
                detail: format!("surface {} links transform {tr}", card.num),
            });
        }
    }

    let mut out = String::from(
        "% Serpent geometry translated from MCNP CSG by nucleide-csg-xlate.\n\
         % Scoped output: surf/cell cards plus a material-name stub only.\n\
         % Supply mat cards (m<n> for MCNP material n), run settings, and\n\
         % an outer boundary yourself; Serpent requires all space defined.\n",
    );
    let mut surfs: Vec<(u32, &'static str, Vec<f64>)> = Vec::with_capacity(deck.surfs.len());
    for card in &deck.surfs {
        let (stype, coeffs) = map_serpent_surface(card)?;
        surfs.push((card.num, stype, coeffs));
    }
    surfs.sort_by_key(|s| s.0);
    // Empty regions (all space) synthesize an `inf` surface.
    let mut inf_id: Option<u32> = None;
    for surf in &surfs {
        let params = surf
            .2
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" ");
        out.push_str(&format!("surf {} {} {}\n", surf.0, surf.1, params));
    }
    for cell in &deck.cells {
        let universe = cell_universe.get(&cell.num).copied().unwrap_or(0);
        let mut region = cell.geom.render();
        if region.trim().is_empty() {
            if inf_id.is_none() {
                let id = ctx.next_id;
                ctx.next_id += 1;
                out.push_str(&format!("surf {id} inf\n"));
                inf_id = Some(id);
            }
            region = format!("-{}", inf_id.expect("set above"));
        }
        let mat = match cell_fill.get(&cell.num) {
            Some(fill) => format!("fill {fill}"),
            None if cell.mat == 0 => "void".to_string(),
            None => format!("m{}", cell.mat),
        };
        out.push_str(&format!(
            "cell {} {} {} {}\n",
            cell.num, universe, mat, region
        ));
    }
    Ok((out, ctx.drift))
}

/// Map one deck surface to a PHITS `[Surface]` line body.
///
/// Returns the PHITS symbol plus parameters in PHITS order. The GO set
/// uses symbols the PHITS manual defines identically to MCNP (`PX/Y/Z`,
/// `SO/SX/SY/SZ/S`, `CX/CY/CZ`, `SPH`, `RPP`, `RCC` including canted
/// axes, axis-aligned `BOX`), so coefficients pass through verbatim.
/// Out-of-scope kinds reuse [`Error::UnsupportedSurface`] and
/// [`Error::MacrobodyOutOfScope`] with PHITS-flavored details.
fn map_phits_surface(card: &SurfCard) -> Result<(&'static str, Vec<f64>)> {
    let c = &card.coeffs;
    match card.kind {
        SurfKind::Px | SurfKind::X => Ok(("PX", vec![c[0]])),
        SurfKind::Py | SurfKind::Y => Ok(("PY", vec![c[0]])),
        SurfKind::Pz | SurfKind::Z => Ok(("PZ", vec![c[0]])),
        SurfKind::So => Ok(("SO", vec![c[0]])),
        SurfKind::Sx => Ok(("SX", vec![c[0], c[1]])),
        SurfKind::Sy => Ok(("SY", vec![c[0], c[1]])),
        SurfKind::Sz => Ok(("SZ", vec![c[0], c[1]])),
        SurfKind::S => Ok(("S", vec![c[0], c[1], c[2], c[3]])),
        SurfKind::Cx => Ok(("CX", vec![c[0]])),
        SurfKind::Cy => Ok(("CY", vec![c[0]])),
        SurfKind::Cz => Ok(("CZ", vec![c[0]])),
        SurfKind::Sph => Ok(("SPH", vec![c[0], c[1], c[2], c[3]])),
        SurfKind::Rpp => Ok(("RPP", vec![c[0], c[1], c[2], c[3], c[4], c[5]])),
        SurfKind::Rcc => Ok(("RCC", vec![c[0], c[1], c[2], c[3], c[4], c[5], c[6]])),
        SurfKind::McBox => Ok((
            "BOX",
            vec![
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11],
            ],
        )),
        SurfKind::P => Err(Error::UnsupportedSurface {
            surf: card.num,
            kind: card.kind.keyword().to_string(),
            detail: "the 9-coefficient three-point plane against PHITS P \
                (origin-side rule) is unverified"
                .to_string(),
        }),
        SurfKind::Kx | SurfKind::Ky | SurfKind::Kz => Err(Error::UnsupportedSurface {
            surf: card.num,
            kind: card.kind.keyword().to_string(),
            detail: "the 5-coefficient cone against PHITS KX/KY/KZ \
                (sqrt-form equation) is unverified"
                .to_string(),
        }),
        SurfKind::Sq => Err(Error::UnsupportedSurface {
            surf: card.num,
            kind: card.kind.keyword().to_string(),
            detail: "the MCNP 10-coefficient SQ ordering against PHITS SQ \
                (A..G plus centre) is unverified"
                .to_string(),
        }),
        SurfKind::Gq => Err(Error::UnsupportedSurface {
            surf: card.num,
            kind: card.kind.keyword().to_string(),
            detail: "the 16-coefficient GQ reduction against PHITS 10-parameter \
                GQ is unverified"
                .to_string(),
        }),
        SurfKind::Tx | SurfKind::Ty | SurfKind::Tz => Err(Error::UnsupportedSurface {
            surf: card.num,
            kind: card.kind.keyword().to_string(),
            detail: "the 6-coefficient torus ordering against PHITS TX/TY/TZ \
                (A/B/C radii) is unverified"
                .to_string(),
        }),
        SurfKind::Rec
        | SurfKind::Wed
        | SurfKind::Rhp
        | SurfKind::Hex
        | SurfKind::Trc
        | SurfKind::Ell
        | SurfKind::Arb => Err(Error::MacrobodyOutOfScope {
            surf: card.num,
            kind: card.kind.keyword().to_string(),
            detail: "phits maps planes, spheres, cylinders, SPH/RPP/RCC/BOX only".to_string(),
        }),
    }
}

/// Translate a parsed MCNP deck to PHITS `[Surface]`/`[Cell]` sections plus drift.
///
/// Same v2 scope as the other directions (surfaces, cells, simple nested
/// universes) with PHITS-native spellings: surface symbols pass through
/// with identical parameters (`RPP`/`RCC`/`BOX` need no expansion, and
/// canted `RCC` maps), `#n` passes through as PHITS's native cell
/// complement, `U=`/`FILL=` render as cell parameters, and `*` markers
/// render as PHITS reflective surfaces. Material cells carry the verbatim
/// MCNP density (positive atom and negative mass densities share PHITS's
/// sign convention); void cells omit it. A void, unfilled cell whose
/// region is a union or complement is emitted as outer void (`-1`, PHITS
/// kills particles there) with an `outer-void-assigned` drift note —
/// review the note when a deck has union-shaped interior voids, for which
/// the heuristic would be wrong. Periodic pointers are loud
/// [`Error::PhitsBoundaryOutOfScope`]s; empty regions are loud (PHITS has
/// no `inf` surface).
pub fn deck_csg_to_phits_input(deck: &DeckProblem) -> Result<(String, DriftTable)> {
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
    let (cell_universe, cell_fill) = resolve_universes(&mut ctx, deck)?;

    // Periodic has no PHITS spelling; `*` markers (card or cell level)
    // force the `*id` reflective surface form.
    let mut reflective: BTreeSet<u32> = BTreeSet::new();
    for card in &deck.surfs {
        if let Some(tr) = card.transform {
            return Err(Error::TransformOutOfScope {
                detail: format!("surface {} links transform {tr}", card.num),
            });
        }
        if card.periodic.is_some() {
            return Err(Error::PhitsBoundaryOutOfScope {
                surf: card.num,
                detail: "periodic surface pointer has no phits spelling".to_string(),
            });
        }
        if card.reflecting {
            reflective.insert(card.num);
        }
    }
    fn collect_reflective(expr: &GeomExpr, out: &mut BTreeSet<u32>) {
        match expr {
            GeomExpr::HalfSpace(h) => {
                if h.reflecting {
                    out.insert(h.surf.unsigned_abs());
                }
            }
            GeomExpr::Intersect(parts) => {
                for part in parts {
                    collect_reflective(part, out);
                }
            }
            GeomExpr::Union(a, b) => {
                collect_reflective(a, out);
                collect_reflective(b, out);
            }
            GeomExpr::Complement(_) => {}
        }
    }
    for cell in &deck.cells {
        collect_reflective(&cell.geom, &mut reflective);
    }

    let mut out = String::from(
        "[ Surface ]\n\
         $ PHITS geometry translated from MCNP CSG by nucleide-csg-xlate.\n\
         $ Scoped output: [Surface]/[Cell] sections plus a material-number\n\
         $ stub only. Supply [Material]/[Parameters] yourself; review any\n\
         $ outer-void-assigned drift notes before running.\n",
    );
    let mut surfs: Vec<(u32, &'static str, Vec<f64>)> = Vec::with_capacity(deck.surfs.len());
    for card in &deck.surfs {
        let (symbol, coeffs) = map_phits_surface(card)?;
        surfs.push((card.num, symbol, coeffs));
    }
    surfs.sort_by_key(|s| s.0);
    for (id, symbol, coeffs) in &surfs {
        let params = coeffs
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" ");
        if reflective.contains(id) {
            out.push_str(&format!("*{id}  {symbol}  {params}\n"));
        } else {
            out.push_str(&format!("{id}  {symbol}  {params}\n"));
        }
    }
    out.push_str("[ Cell ]\n");
    for cell in &deck.cells {
        let region = cell.geom.render().replace('*', "");
        if region.trim().is_empty() {
            return Err(Error::UniverseOutOfScope {
                detail: format!(
                    "cell {} has an empty region (no phits inf spelling)",
                    cell.num
                ),
            });
        }
        let is_fill = cell_fill.contains_key(&cell.num);
        let mat = if is_fill {
            // Filled cells carry a material number PHITS ignores; use void.
            0
        } else {
            cell.mat
        };
        // Outer-void heuristic: void, unfilled, union/complement-shaped.
        let outer = mat == 0 && (region.contains(':') || region.contains('#'));
        let mat_field = if outer { -1 } else { mat as i32 };
        if outer {
            ctx.note(
                DriftScope::Cell,
                cell.num,
                "outer-void-assigned",
                format!(
                    "cell {num} void union/complement emitted as outer void -1",
                    num = cell.num
                ),
            );
        }
        let mut line = if mat_field == 0 || mat_field == -1 {
            format!("{}  {mat_field}  {region}", cell.num)
        } else {
            let dens = cell.dens.unwrap_or(0.0);
            format!("{}  {mat_field}  {dens}  {region}", cell.num)
        };
        if let Some(universe) = cell_universe.get(&cell.num) {
            if *universe != 0 {
                line.push_str(&format!("  U={universe}"));
            }
        }
        if let Some(fill) = cell_fill.get(&cell.num) {
            line.push_str(&format!("  FILL={fill}"));
        }
        line.push('\n');
        out.push_str(&line);
    }
    Ok((out, ctx.drift))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nucleide_mcnp_io::problem::parse_deck;

    fn translate(text: &str) -> Result<(String, DriftTable)> {
        let deck = parse_deck(text).unwrap();
        deck_csg_to_openmc_xml(&deck)
    }

    fn deck_text(cells: &str, surfs: &str, data: &str) -> String {
        format!("msg\ntitle\n{cells}\n\n{surfs}\n\n{data}\n")
    }

    #[test]
    fn sphere_and_planes_map_lossless() {
        let (xml, drift) = translate(&deck_text(
            "1 1 -1.0 -1\n2 0 1 2 -3 4",
            "1 so 5.0\n2 px -10.0\n3 px 10.0\n4 py -10.0",
            "mode n\nm1 92235 1.0",
        ))
        .unwrap();
        assert!(xml.contains("<surface id=\"1\" type=\"sphere\" coeffs=\"0 0 0 5\""));
        assert!(xml.contains("<surface id=\"2\" type=\"x-plane\" coeffs=\"-10\""));
        assert!(xml.contains("<cell id=\"1\" universe=\"0\" material=\"1\" region=\"-1\""));
        assert!(xml.contains("region=\"1 2 -3 4\""));
        assert!(xml.contains("material=\"void\""));
        // MODE is a dropped data card; nothing else drifts.
        assert_eq!(drift.len(), 1);
        assert_eq!(drift.entries[0].action, "dropped-data-card");
    }

    #[test]
    fn axis_cylinders_and_sphere_aliases() {
        let (xml, _) = translate(&deck_text(
            "1 1 -1.0 -1 2 -3",
            "1 sx 4 5\n2 cy 2\n3 cz 3",
            "m1 1001 1.0",
        ))
        .unwrap();
        assert!(xml.contains("type=\"sphere\" coeffs=\"4 0 0 5\""));
        assert!(xml.contains("type=\"y-cylinder\" coeffs=\"0 0 2\""));
        assert!(xml.contains("type=\"z-cylinder\" coeffs=\"0 0 3\""));
        assert!(xml.contains("region=\"-1 2 -3\""));
    }

    #[test]
    fn rpp_expands_to_six_planes() {
        let (xml, drift) = translate(&deck_text(
            "1 1 -1.0 -1\n2 0 1",
            "1 rpp -5 5 -5 5 -5 5",
            "m1 13027 1.0",
        ))
        .unwrap();
        // Facet ids start above the deck maximum (1 -> 2..7).
        assert!(xml.contains("<surface id=\"2\" type=\"x-plane\" coeffs=\"-5\""));
        assert!(xml.contains("<surface id=\"3\" type=\"x-plane\" coeffs=\"5\""));
        assert!(xml.contains("region=\"2 -3 4 -5 6 -7\""));
        assert!(xml.contains("region=\"-2 | 3 | -4 | 5 | -6 | 7\""));
        assert!(drift
            .for_scope(DriftScope::Surface)
            .iter()
            .any(|e| e.action == "macrobody-expansion"));
    }

    #[test]
    fn rcc_axis_aligned_expands() {
        let (xml, drift) = translate(&deck_text(
            "1 1 -1.0 -1\n2 0 1",
            "1 rcc 0 0 -5 0 0 10 2",
            "m1 1001 1.0",
        ))
        .unwrap();
        assert!(xml.contains("type=\"z-cylinder\" coeffs=\"0 0 2\""));
        assert!(xml.contains("region=\"-2 3 -4\""));
        assert!(drift
            .for_scope(DriftScope::Surface)
            .iter()
            .any(|e| e.action == "macrobody-expansion"));
    }

    #[test]
    fn rcc_canted_is_out_of_scope() {
        let err = translate(&deck_text(
            "1 1 -1.0 -1",
            "1 rcc 0 0 0 1 1 1 2",
            "m1 1001 1.0",
        ))
        .unwrap_err();
        assert!(matches!(err, Error::MacrobodyOutOfScope { .. }), "{err}");
    }

    #[test]
    fn cones_quadrics_tori_planes_rejected() {
        for (kind, coeffs) in [
            ("kx", "0 0 0 1 1"),
            ("sq", "1 1 1 0 0 0 0 0 0 -25"),
            ("gq", "1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 -25"),
            ("tx", "0 0 0 10 1 1"),
            ("p", "0 0 0 1 0 0 0 1 0"),
            ("box", "0 0 0 1 0 0 0 1 0 0 0 1"),
            ("arb", "0 0 0"),
        ] {
            let err = translate(&deck_text(
                "1 1 -1.0 -1",
                &format!("1 {kind} {coeffs}"),
                "m1 1001 1.0",
            ))
            .unwrap_err();
            assert!(
                matches!(
                    err,
                    Error::UnsupportedSurface { .. } | Error::MacrobodyOutOfScope { .. }
                ),
                "{kind}: {err}"
            );
        }
    }

    #[test]
    fn simple_complement_inlines() {
        let (xml, drift) = translate(&deck_text(
            "1 1 -1.0 -1\n2 0 #1 -2",
            "1 so 10\n2 pz 0",
            "m1 92235 1.0",
        ))
        .unwrap();
        assert!(xml.contains("region=\"1 -2\""));
        assert!(drift
            .for_scope(DriftScope::Cell)
            .iter()
            .any(|e| e.action == "complement-expansion"));
    }

    #[test]
    fn union_complement_is_too_complex() {
        let err = translate(&deck_text(
            "1 1 -1.0 -1 : -2\n2 0 #1",
            "1 so 10\n2 so 20",
            "m1 92235 1.0",
        ))
        .unwrap_err();
        assert!(matches!(err, Error::ComplementTooComplex { .. }), "{err}");
    }

    #[test]
    fn nested_complement_is_too_complex() {
        let err = translate(&deck_text(
            "1 1 -1.0 -1\n2 0 #1 -2\n3 0 #2",
            "1 so 10\n2 so 20",
            "m1 92235 1.0",
        ))
        .unwrap_err();
        assert!(matches!(err, Error::ComplementTooComplex { .. }), "{err}");
    }

    #[test]
    fn complement_of_missing_cell_errors() {
        // Deck validation rejects the dangling complement first (Mcnp wrap).
        let err = translate(&deck_text("1 0 #9", "1 so 10", "")).unwrap_err();
        assert!(matches!(err, Error::Mcnp(..)), "{err}");
    }

    #[test]
    fn reflecting_surface_card_sets_boundary() {
        let (xml, _) = translate(&deck_text("1 0 -1", "*1 so 10", "")).unwrap();
        assert!(xml.contains("boundary=\"reflective\""));
    }

    #[test]
    fn cell_level_reflecting_forces_surface() {
        let (xml, drift) = translate(&deck_text("1 0 *-1 2", "1 so 10\n2 so 20", "")).unwrap();
        assert!(xml.contains(
            "<surface id=\"1\" type=\"sphere\" coeffs=\"0 0 0 10\" boundary=\"reflective\""
        ));
        // Marker stripped from the region, surface forced reflective.
        assert!(xml.contains("region=\"-1 2\""));
        assert!(drift
            .for_scope(DriftScope::Cell)
            .iter()
            .any(|e| e.action == "reflective-applied"));
    }

    #[test]
    fn reflective_periodic_conflict_errors() {
        let err = translate(&deck_text("1 0 -1", "*1 -2 pz 0\n2 pz 5", "")).unwrap_err();
        assert!(matches!(err, Error::ReflectiveConflict { .. }), "{err}");
    }

    #[test]
    fn periodic_pair_links() {
        let (xml, drift) = translate(&deck_text("1 0 -1 2", "1 -2 pz 0\n2 pz 5", "")).unwrap();
        assert!(xml.contains("boundary=\"periodic\" periodic_surface_id=\"2\""));
        assert!(xml.contains("boundary=\"periodic\" periodic_surface_id=\"1\""));
        assert!(drift
            .for_scope(DriftScope::Surface)
            .iter()
            .any(|e| e.action == "periodic-link"));
    }

    #[test]
    fn periodic_divergent_partner_is_ambiguous() {
        let err = translate(&deck_text("1 0 -1", "1 -2 pz 0\n2 -3 pz 5\n3 pz 9", "")).unwrap_err();
        assert!(matches!(err, Error::PeriodicAmbiguous { .. }), "{err}");
    }

    #[test]
    fn periodic_missing_partner_is_ambiguous() {
        // Deck validation rejects the dangling partner first (Mcnp wrap).
        let err = translate(&deck_text("1 0 -1", "1 -9 pz 0", "")).unwrap_err();
        assert!(matches!(err, Error::Mcnp(..)), "{err}");
    }

    #[test]
    fn lattices_and_matrix_fills_error() {
        // Parameters chosen to pass deck validation (universe 0 exists, TR1
        // exists) so the scope errors fire instead of Mcnp link errors.
        // Plain `u=0` and single `fill=0` translate under v2 (tested below).
        for (params, data) in [("lat=1 fill=0", ""), ("u=-1", ""), ("trcl=1", "tr1 0 0 0")] {
            let err =
                translate(&deck_text(&format!("1 0 -1 {params}"), "1 so 10", data)).unwrap_err();
            assert!(
                matches!(
                    err,
                    Error::UniverseOutOfScope { .. } | Error::TransformOutOfScope { .. }
                ),
                "{params}: {err}"
            );
        }
    }

    #[test]
    fn nested_universe_fill_translates() {
        // Universe 1 holds the sphere; universe 0 fills it from the box cell.
        let (xml, drift) = translate(&deck_text(
            "1 1 -1.0 -1 u=1\n2 0 -2 fill=1",
            "1 so 5\n2 so 10",
            "m1 92235 1.0",
        ))
        .unwrap();
        assert!(xml.contains("universe=\"1\""));
        assert!(xml.contains("fill=\"1\""));
        // A filled cell carries no material attribute.
        assert!(!xml.contains("fill=\"1\" material="));
        assert!(xml.contains("material=\"1\""));
        assert!(drift
            .for_scope(DriftScope::Cell)
            .iter()
            .any(|e| e.action == "universe-assigned"));
        assert!(drift
            .for_scope(DriftScope::Cell)
            .iter()
            .any(|e| e.action == "fill-applied"));
    }

    #[test]
    fn tallies_sources_transforms_error() {
        for data in [
            "sdef pos=0 0 0",
            "kcode 1000 1.0 10 100",
            "f4:n 1",
            "tr1 0 0 5",
        ] {
            let err = translate(&deck_text("1 0 -1", "1 so 10", data)).unwrap_err();
            assert!(
                matches!(
                    err,
                    Error::TallySourceOutOfScope { .. } | Error::TransformOutOfScope { .. }
                ),
                "{data}: {err}"
            );
        }
    }

    #[test]
    fn unknown_surface_reference_errors() {
        // Deck validation rejects the dangling reference first (Mcnp wrap).
        let err = translate(&deck_text("1 0 -9", "1 so 10", "")).unwrap_err();
        assert!(matches!(err, Error::Mcnp(..)), "{err}");
    }

    #[test]
    fn drift_table_helpers() {
        let table = DriftTable::default();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert!(table.for_scope(DriftScope::Cell).is_empty());
    }

    fn translate_serpent(text: &str) -> Result<(String, DriftTable)> {
        use nucleide_mcnp_io::problem::parse_deck;
        deck_csg_to_serpent_input(&parse_deck(text).unwrap())
    }

    fn translate_phits(text: &str) -> Result<(String, DriftTable)> {
        use nucleide_mcnp_io::problem::parse_deck;
        deck_csg_to_phits_input(&parse_deck(text).unwrap())
    }
    #[test]
    fn serpent_sphere_box_cards() {
        let (text, _) = translate_serpent(&deck_text(
            "1 1 -1.0 -1\n2 0 -2",
            "1 so 5\n2 so 10",
            "m1 92235 1.0",
        ))
        .unwrap();
        assert!(text.contains("surf 1 sph 0 0 0 5\n"));
        assert!(text.contains("surf 2 sph 0 0 0 10\n"));
        assert!(text.contains("cell 1 0 m1 -1\n"));
        assert!(text.contains("cell 2 0 void -2\n"));
    }

    #[test]
    fn serpent_rpp_is_native_cuboid() {
        let (text, drift) = translate_serpent(&deck_text(
            "1 1 -1.0 -1",
            "1 rpp -1 1 -2 2 -3 3",
            "m1 92235 1.0",
        ))
        .unwrap();
        assert!(text.contains("surf 1 cuboid -1 1 -2 2 -3 3\n"));
        assert!(!drift
            .entries
            .iter()
            .any(|e| e.action == "macrobody-expansion"));
    }

    #[test]
    fn serpent_universe_fill_cards() {
        let (text, drift) = translate_serpent(&deck_text(
            "1 1 -1.0 -1 u=1\n2 0 -2 fill=1",
            "1 so 5\n2 so 10",
            "m1 92235 1.0",
        ))
        .unwrap();
        assert!(text.contains("cell 1 1 m1 -1\n"));
        assert!(text.contains("cell 2 0 fill 1 -2\n"));
        assert!(drift.entries.iter().any(|e| e.action == "fill-applied"));
    }

    #[test]
    fn serpent_complement_passes_through() {
        let (text, _) = translate_serpent(&deck_text(
            "1 1 -1.0 -1\n2 0 #1 -2",
            "1 so 5\n2 so 10",
            "m1 92235 1.0",
        ))
        .unwrap();
        assert!(text.contains("cell 2 0 void #1 -2\n"));
    }

    #[test]
    fn serpent_boundaries_are_loud() {
        let err = translate_serpent(&deck_text("1 0 -1", "*1 so 10", "")).unwrap_err();
        assert!(
            matches!(err, Error::SerpentBoundaryOutOfScope { .. }),
            "{err}"
        );
        let err = translate_serpent(&deck_text("1 0 *-1", "1 so 10", "")).unwrap_err();
        assert!(
            matches!(err, Error::SerpentBoundaryOutOfScope { .. }),
            "{err}"
        );
        let err = translate_serpent(&deck_text("1 0 -1 2", "1 -2 pz 0\n2 pz 5", "")).unwrap_err();
        assert!(
            matches!(err, Error::SerpentBoundaryOutOfScope { .. }),
            "{err}"
        );
        let err = translate_serpent(&deck_text("1 0 -1", "1 kz 0 0 0 1 1", "")).unwrap_err();
        assert!(matches!(err, Error::UnsupportedSurface { .. }), "{err}");
    }

    #[test]
    fn errors_are_non_exhaustive_and_display() {
        let err = Error::ComplementTooComplex { cell: 2, target: 1 };
        assert!(err.to_string().contains("cell 2 complement #1"));
        let err = Error::UniverseOutOfScope {
            detail: "x".to_string(),
        };
        assert!(err.to_string().contains("universes/lattices/fills"));
        let err = Error::SerpentBoundaryOutOfScope {
            surf: 1,
            detail: "x".to_string(),
        };
        assert!(err.to_string().contains("no verified serpent mapping"));
        let err = Error::PhitsBoundaryOutOfScope {
            surf: 1,
            detail: "x".to_string(),
        };
        assert!(err.to_string().contains("no phits spelling"));
    }

    #[test]
    fn phits_sphere_box_sections() {
        let (text, _) = translate_phits(&deck_text(
            "1 1 -1.0 -1\n2 0 -2",
            "1 so 5\n2 so 10",
            "m1 92235 1.0",
        ))
        .unwrap();
        assert!(text.contains("[ Surface ]\n"));
        assert!(text.contains("[ Cell ]\n"));
        assert!(text.contains("1  SO  5\n"));
        assert!(text.contains("2  SO  10\n"));
        assert!(text.contains("1  1  -1  -1\n"));
        // Void without union/complement stays material 0 (density omitted).
        assert!(text.contains("2  0  -2\n"));
    }

    #[test]
    fn phits_outer_void_and_fill() {
        let (text, drift) = translate_phits(&deck_text(
            "1 1 -1.0 -1 u=1\n2 0 -2 fill=1",
            "1 so 5\n2 so 10",
            "m1 92235 1.0",
        ))
        .unwrap();
        assert!(text.contains("1  1  -1  -1  U=1\n"));
        assert!(text.contains("2  0  -2  FILL=1\n"));
        assert!(drift.entries.iter().any(|e| e.action == "fill-applied"));
        // Complement-shaped void becomes outer void -1.
        let (text, drift) = translate_phits(&deck_text(
            "1 1 -1.0 -1\n2 0 #1 -2",
            "1 so 5\n2 so 10",
            "m1 92235 1.0",
        ))
        .unwrap();
        assert!(text.contains("2  -1  #1 -2\n"));
        assert!(drift
            .entries
            .iter()
            .any(|e| e.action == "outer-void-assigned"));
    }

    #[test]
    fn phits_reflective_maps_and_periodic_is_loud() {
        let (text, _) = translate_phits(&deck_text("1 0 -1", "*1 so 10", "")).unwrap();
        assert!(text.contains("*1  SO  10\n"));
        let (text, _) = translate_phits(&deck_text("1 0 *-1", "1 so 10", "")).unwrap();
        assert!(text.contains("*1  SO  10\n"));
        let err = translate_phits(&deck_text("1 0 -1 2", "1 -2 pz 0\n2 pz 5", "")).unwrap_err();
        assert!(
            matches!(err, Error::PhitsBoundaryOutOfScope { .. }),
            "{err}"
        );
        let err = translate_phits(&deck_text("1 0 -1", "1 kz 0 0 0 1 1", "")).unwrap_err();
        assert!(matches!(err, Error::UnsupportedSurface { .. }), "{err}");
    }

    #[test]
    fn phits_canted_rcc_maps() {
        // PHITS RCC takes an arbitrary height vector natively.
        let (text, _) = translate_phits(&deck_text(
            "1 1 -1.0 -1",
            "1 rcc 0 0 0 1 1 1 2",
            "m1 92235 1.0",
        ))
        .unwrap();
        assert!(text.contains("1  RCC  0 0 0 1 1 1 2\n"));
    }
}
