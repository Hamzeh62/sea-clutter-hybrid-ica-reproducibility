"""Corrected unit sampler with enforced target/guard exclusion (Phase 4A).

Drop-in replacement for ``src/eval/units.py::UnitSampler``.  Differences:

* acquisition identity comes from :mod:`acquisition_ids` (strict canonical ID),
* a missing or empty exclusion set is a hard error,
* every sampled CUT cell is asserted against the forbidden set,
* the range dimension of the cached cube must equal the metadata length,
* the background policy for CFAR training cells and ANMF secondary windows is
  explicit (``clean`` = target/guard cells removed, ``operational`` = all cells
  except the CUT, which is what a deployed detector actually sees).

The sampler never writes to the cache or to the source project.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from acquisition_ids import (AcquisitionRecord, assert_cell_selection,
                             canonical_acq_id)


class FixedUnitSampler:
    def __init__(self, split_files: list[str], Np: int, hop: int,
                 table: dict[str, AcquisitionRecord], pol: str = "VV",
                 cross_pol: str | None = "VH", background_policy: str = "clean"):
        if background_policy not in ("clean", "operational"):
            raise ValueError("background_policy must be 'clean' or 'operational'")
        self.Np, self.hop, self.pol = int(Np), int(hop), pol
        self.table = table
        self.background_policy = background_policy
        self.units: list[tuple[str, int, int]] = []
        self.data: dict[str, np.ndarray] = {}
        self.data_cross: dict[str, np.ndarray] = {}
        self.allowed_cells: dict[str, tuple[int, ...]] = {}
        self.forbidden_cells: dict[str, tuple[int, ...]] = {}
        self.n_units_by_file: dict[str, int] = {}

        for f in split_files:
            aid = canonical_acq_id(f)
            if aid not in table:
                raise KeyError(f"{aid}: no canonical metadata record (refusing silent default)")
            rec = table[aid]
            z = np.load(f)
            if pol not in z:
                raise KeyError(f"{aid}: polarization {pol} absent from {f}")
            cube = z[pol]
            ncell = int(cube.shape[1])
            if ncell != rec.n_range:
                raise ValueError(
                    f"{aid}: cache cube has {ncell} range cells but metadata has {rec.n_range}"
                )
            forbidden = set(rec.forbidden)
            if not forbidden:
                raise ValueError(f"{aid}: empty exclusion set")
            n_win = (cube.shape[0] - self.Np) // self.hop + 1
            if n_win <= 0:
                raise ValueError(f"{aid}: cube shorter than one CPI")
            for c in range(ncell):
                if c in forbidden:
                    continue
                for w in range(n_win):
                    self.units.append((aid, c, w * self.hop))
            self.data[aid] = cube
            self.data_cross[aid] = z[cross_pol] if cross_pol else cube
            self.allowed_cells[aid] = tuple(c for c in range(ncell) if c not in forbidden)
            self.forbidden_cells[aid] = tuple(sorted(forbidden))
            self.n_units_by_file[aid] = len(self.allowed_cells[aid]) * n_win

            selected = {c for k, c, w in self.units if k == aid}
            assert_cell_selection(aid, selected, table)
            if len(selected) != len(self.allowed_cells[aid]):
                raise AssertionError(f"{aid}: not every eligible cell was sampled")

        if not self.units:
            raise RuntimeError("no units selected; refusing an empty experiment")

    def __len__(self) -> int:
        return len(self.units)

    def draw(self, rng: np.random.Generator) -> tuple[str, int, int]:
        return self.units[int(rng.integers(len(self.units)))]

    def y_cpi(self, key: str, cell: int, w0: int, inject=None) -> np.ndarray:
        y = self.data[key][w0:w0 + self.Np, cell].astype(np.complex128)
        return inject(y) if inject else y

    def _background_cells(self, key: str, lo: int, hi: int) -> list[int]:
        if self.background_policy == "clean":
            allowed = set(self.allowed_cells[key])
            return [c for c in range(lo, hi) if c in allowed]
        return list(range(lo, hi))

    def range_block(self, key: str, cell: int, w0: int, half: int,
                    inject_cell: int | None = None, inject=None):
        """Contiguous range block with forbidden cells removed under 'clean'."""
        cube = self.data[key]
        ncell = cube.shape[1]
        lo, hi = max(0, cell - half), min(ncell, cell + half + 1)
        cells_idx = self._background_cells(key, lo, hi)
        if cell not in cells_idx:
            raise AssertionError(f"{key}: CUT cell {cell} missing from its own range block")
        cells = [cube[w0:w0 + self.Np, c].astype(np.complex128) for c in cells_idx]
        ci = cells_idx.index(cell)
        tgt = cell if inject_cell is None else inject_cell
        if inject is not None and tgt in cells_idx:
            cells[cells_idx.index(tgt)] = inject(cells[cells_idx.index(tgt)])
        return cells, ci

    def secondary_windows(self, key: str, cell: int, w0: int, K: int,
                          cells_each_side: int = 4) -> np.ndarray:
        """K unit-norm secondary windows, policy-controlled background cells."""
        cube = self.data[key]
        ncell = cube.shape[1]
        lo, hi = max(0, cell - cells_each_side), min(ncell, cell + cells_each_side + 1)
        cand_cells = [c for c in self._background_cells(key, lo, hi) if c != cell]
        if not cand_cells:
            return np.empty((0, self.Np), np.complex128)
        per_cell = int(np.ceil(K / len(cand_cells)))
        offs = np.arange(per_cell) - per_cell // 2
        wins: list[np.ndarray] = []
        for c in cand_cells:
            for o in offs:
                s = w0 + int(o) * self.hop
                if s < 0 or s + self.Np > cube.shape[0]:
                    continue
                wins.append(cube[s:s + self.Np, c].astype(np.complex128))
                if len(wins) >= K:
                    return np.stack(wins)
        return np.stack(wins[:K]) if wins else np.empty((0, self.Np), np.complex128)

    def protocol_manifest(self) -> dict:
        return {
            "background_policy": self.background_policy,
            "Np": self.Np, "hop": self.hop, "pol": self.pol,
            "files": {
                aid: {"allowed_cells": list(self.allowed_cells[aid]),
                      "forbidden_cells": list(self.forbidden_cells[aid]),
                      "n_units": self.n_units_by_file[aid]}
                for aid in sorted(self.allowed_cells)
            },
            "n_units_total": len(self.units),
        }
