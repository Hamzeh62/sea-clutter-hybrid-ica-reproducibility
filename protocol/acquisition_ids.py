"""Canonical acquisition identity and target/guard-cell exclusion (Phase 4A).

The 2026 Phase-3 audit found that `src/eval/units.py` built cache keys as
`Path(f).stem` (``..._cubes``) but looked up ``target_cells.json`` keys after
removing only ``.cdf`` (``..._starea``).  Every lookup returned ``None``, the
exclusion set was silently empty, and 38 % of the test cells the protocol meant
to forbid were sampled.

This module replaces that fragile matching with one canonical acquisition ID:

    raw CDF   19931107_141630_starea.cdf
    cache     19931107_141630_starea_cubes.npz
    metadata  19931107_141630_starea.cdf
    -> acq_id 19931107_141630_starea

Rules
-----
* The ID is parsed from a strict regex after repeatedly stripping the known
  transport suffixes (``_cubes``, ``.npz``, ``.cdf``, ``.CDF``).  No substring
  matching, no ``replace`` on arbitrary text.
* Site names are case-folded (``stareC0000`` -> ``starec0000``); the date/time
  fields are kept verbatim.
* Duplicate IDs, unknown suffixes, missing metadata and empty exclusion sets
  are hard errors, never silent defaults.
* Cell indices in ``target_cells.json`` are 0-based cube columns.  Literature
  target cells (D43) are 1-based range-gate numbers; the 0-based equivalent is
  ``literature - 1``.  This is cross-validated in ``tests/test_acquisition_ids``.

Exclusion policy
----------------
``targets`` = statistical candidates (``power_zscore > 1.0``) UNION the
literature cell when one exists.  ``forbidden`` = every target cell plus its
two immediate range neighbours (the paper's "target + one guard" rule), clipped
to ``[0, n_range)``.  When the two sources disagree the cell is still excluded
and the acquisition is flagged ``REVIEW`` rather than silently resolved.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

CANON_RE = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<site>[A-Za-z][A-Za-z0-9]*)$")
TRANSPORT_SUFFIXES = (".npz", ".cdf", ".CDF", "_cubes")
Z_THRESHOLD = 1.0

# D43 literature target cells, 1-based as published (see d43_real_target.py).
LITERATURE_TARGET_CELLS_1BASED: dict[str, tuple[int, str]] = {
    "19931107_135603_starea.cdf": (9, "rs14_4364;iet2017"),
    "19931107_141630_starea.cdf": (9, "sensors22_9163"),
    "19931107_145028_starea.cdf": (8, "rs14_4364;iet2017"),
    "19931108_213827_starea.cdf": (7, "rs14_4364;iet2017"),
    "19931108_220902_starea.cdf": (7, "sensors22_9163;iet2017"),
    "19931109_191449_starea.cdf": (7, "sensors22_9163;iet2017"),
    "19931109_202217_starea.cdf": (7, "iet2017"),
    "19931110_001635_starea.cdf": (7, "iet2017"),
}


def canonical_acq_id(name_or_path: str | Path) -> str:
    """Map a raw filename, cache filename or metadata key to one acquisition ID."""
    base = Path(str(name_or_path)).name
    changed = True
    while changed:
        changed = False
        for suf in TRANSPORT_SUFFIXES:
            if base.endswith(suf):
                base = base[: -len(suf)]
                changed = True
    m = CANON_RE.match(base)
    if not m:
        raise ValueError(
            f"cannot canonicalise acquisition name {name_or_path!r}: "
            f"residual {base!r} does not match {CANON_RE.pattern}"
        )
    return f"{m.group('date')}_{m.group('time')}_{m.group('site').lower()}"


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class AcquisitionRecord:
    acq_id: str
    raw_name: str
    cache_name: str
    split_role: str
    n_range: int
    candidates: tuple[int, ...]
    literature_1based: int | None
    literature_0based: int | None
    authoritative: int
    targets: tuple[int, ...]
    forbidden: tuple[int, ...]
    eligible: tuple[int, ...]
    z_literature: float | None
    status: str
    notes: str

    @property
    def n_forbidden(self) -> int:
        return len(self.forbidden)

    @property
    def n_eligible(self) -> int:
        return len(self.eligible)


def _build_record(acq_id: str, raw_name: str, cache_name: str, role: str,
                  z: list[float]) -> AcquisitionRecord:
    n_range = len(z)
    if n_range <= 2:
        raise ValueError(f"{acq_id}: implausible range dimension {n_range}")
    candidates = tuple(int(i) for i in range(n_range) if z[i] > Z_THRESHOLD)
    if not candidates:
        raise ValueError(f"{acq_id}: no statistical target candidate above z={Z_THRESHOLD}")

    lit = LITERATURE_TARGET_CELLS_1BASED.get(raw_name)
    lit0 = lit[0] - 1 if lit else None
    z_lit = float(z[lit0]) if lit0 is not None else None
    if lit0 is not None and not (0 <= lit0 < n_range):
        raise ValueError(f"{acq_id}: literature cell {lit[0]} outside [1,{n_range}]")

    targets = sorted(set(candidates) | ({lit0} if lit0 is not None else set()))
    forbidden: set[int] = set()
    for c in targets:
        forbidden.update({c - 1, c, c + 1})
    forbidden = {c for c in forbidden if 0 <= c < n_range}
    if not forbidden:
        raise ValueError(f"{acq_id}: empty forbidden set after clipping")
    if len(forbidden) == n_range:
        raise ValueError(f"{acq_id}: exclusion would forbid every range cell")

    authoritative = lit0 if lit0 is not None else int(max(range(n_range), key=lambda i: z[i]))
    notes = []
    status = "OK"
    if lit0 is not None and lit0 not in candidates:
        status = "REVIEW_LITERATURE_STAT_MISMATCH"
        notes.append(
            f"literature cell {lit[0]} (0-based {lit0}) has z={z_lit:.2f} but statistical "
            f"candidates are {list(candidates)}; both are excluded"
        )
    if len(candidates) > 1:
        notes.append(f"{len(candidates)} statistical candidates (target range extent)")
    if lit0 is None:
        notes.append("no literature cell; authoritative = argmax power z-score")

    return AcquisitionRecord(
        acq_id=acq_id, raw_name=raw_name, cache_name=cache_name, split_role=role,
        n_range=n_range, candidates=candidates, literature_1based=lit[0] if lit else None,
        literature_0based=lit0, authoritative=int(authoritative), targets=tuple(targets),
        forbidden=tuple(sorted(forbidden)),
        eligible=tuple(c for c in range(n_range) if c not in forbidden),
        z_literature=z_lit, status=status, notes="; ".join(notes),
    )


def build_table(project_root: str | Path, cache_dir: str | Path,
                split_roles: dict[str, str] | None = None) -> dict[str, AcquisitionRecord]:
    """Build the canonical acquisition table and reject every ambiguous mapping."""
    project_root = Path(project_root)
    cache_dir = Path(cache_dir)
    # Release layout: ``metadata/target_cells.json`` at the repository root.  The
    # historical location is still accepted, so the module can be pointed at an
    # untouched checkout of the original project.
    meta_path = Path(project_root) / "metadata/target_cells.json"
    if not meta_path.is_file():
        meta_path = Path(project_root) / "ipix_hybrid_detector/results/json/target_cells.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"no target_cells.json under {project_root} (looked in metadata/ and "
            f"ipix_hybrid_detector/results/json/)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    cache_by_id: dict[str, Path] = {}
    for p in sorted(cache_dir.glob("*_cubes.npz")):
        aid = canonical_acq_id(p)
        if aid in cache_by_id:
            raise ValueError(f"duplicate cache acquisition ID {aid}")
        cache_by_id[aid] = p

    table: dict[str, AcquisitionRecord] = {}
    for raw_name, info in sorted(meta.items()):
        aid = canonical_acq_id(raw_name)
        if aid in table:
            raise ValueError(f"duplicate metadata acquisition ID {aid}")
        if aid not in cache_by_id:
            raise FileNotFoundError(f"{aid}: metadata present but no cache cube in {cache_dir}")
        role = (split_roles or {}).get(aid, "unassigned")
        table[aid] = _build_record(aid, raw_name, cache_by_id[aid].name, role,
                                   info["power_zscore"])

    missing_meta = sorted(set(cache_by_id) - set(table))
    if missing_meta:
        raise ValueError(f"cache acquisitions without target metadata: {missing_meta}")
    return table


def assert_cell_selection(acq_id: str, cells, table: dict[str, AcquisitionRecord]) -> None:
    """Raise if a sampled cell set intersects the forbidden set of an acquisition."""
    rec = table[acq_id]
    selected = {int(c) for c in cells}
    bad = selected & set(rec.forbidden)
    if bad:
        raise AssertionError(
            f"{acq_id}: selection contains forbidden target/guard cells {sorted(bad)}"
        )
    out = selected - set(range(rec.n_range))
    if out:
        raise AssertionError(f"{acq_id}: cells outside [0,{rec.n_range}): {sorted(out)}")


def write_map_csv(table: dict[str, AcquisitionRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "acq_id", "raw_name", "cache_name", "split_role", "n_range",
            "statistical_candidates_0based", "literature_1based", "literature_0based",
            "authoritative_0based", "target_cells_0based", "forbidden_cells_0based",
            "n_forbidden", "n_eligible", "eligible_cells_0based", "z_at_literature",
            "status", "notes",
        ])
        for aid in sorted(table):
            r = table[aid]
            w.writerow([
                r.acq_id, r.raw_name, r.cache_name, r.split_role, r.n_range,
                ";".join(map(str, r.candidates)),
                "" if r.literature_1based is None else r.literature_1based,
                "" if r.literature_0based is None else r.literature_0based,
                r.authoritative, ";".join(map(str, r.targets)),
                ";".join(map(str, r.forbidden)), r.n_forbidden, r.n_eligible,
                ";".join(map(str, r.eligible)),
                "" if r.z_literature is None else f"{r.z_literature:.3f}",
                r.status, r.notes,
            ])


def table_as_json(table: dict[str, AcquisitionRecord]) -> dict:
    return {aid: {**r.__dict__,
                  "candidates": list(r.candidates), "targets": list(r.targets),
                  "forbidden": list(r.forbidden), "eligible": list(r.eligible)}
            for aid, r in sorted(table.items())}
