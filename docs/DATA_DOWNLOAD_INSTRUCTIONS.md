# Obtaining the IPIX recordings

The radar data are **not** distributed with this repository, and this document deliberately does not
invent a download link. The IPIX database is a third-party research dataset distributed by its
original provider on physical media and, in some periods, through academic mirrors; the authoritative
route at any given time is the one published by that provider.

## 1. What to obtain

Sixteen specific recordings are used. Their exact names are the keys of
`../metadata/target_cells.json`:

```
19931107_135603_starea.cdf          19931107_141630_starea.cdf
19931107_145028_starea.cdf          19931108_213827_starea.cdf
19931108_220902_starea.cdf          19931109_191449_starea.cdf
19931109_202217_starea.cdf          19931110_001635_starea.cdf
19931111_163625_starea.cdf          19931118_023604_stareC0000.cdf
19931118_035737_stareC0000.cdf      19931118_162155_stareC0000.cdf
19931118_162658_stareC0000.cdf      19931118_174259_stareC0000.cdf
19980204_163113_ANTSTEP.CDF         19980205_171437_ANTSTEP.CDF
```

The author's copy came from the standard IPIX distribution, whose media are labelled
`IPIX_CD1.ISO` … `IPIX_CD12.ISO` (11 volumes are present in the author's copy; the recordings above
are spread across them). If you have those volumes, mount or extract them and locate the `.cdf`
files by name. Other institutions mirror the same database; use whichever mirror your library
confirms as legitimate.

Do not use a mirror that requires you to accept terms incompatible with research use, and do not
upload the recordings to a public repository.

## 2. Verify you have the right files before using them

The reader accepts the standard IPIX NetCDF layout. A quick check on the source files, without
building anything:

```python
import sys; sys.path.insert(0, ".")
from src.data.ipix_loader import read_ipix_file
rec = read_ipix_file("19931107_135603_starea.cdf")
print(rec["attrs"]["PRF"], rec["data"]["VV"].shape)     # 1000.0  (131072, 14)
```

Expected dimensions per campaign:

| campaign | recordings | VV shape (pulses, range cells) | PRF |
|---|---|---|---|
| 1993 `starea` / `stareC0000` | 14 | (131072, 14) | 1000 Hz |
| 1998 `ANTSTEP` | 2 | (60000, 28) | 1000 Hz |

A file with a different shape is a different acquisition or a different processing level; do not
substitute it. The two `stareC0000` acquisitions are the "C0000" beam-position variant and keep the
`c0000` suffix in their canonical ID (`protocol/acquisition_ids.py`), which is what the partition
manifest uses.

## 3. Build and verify the cache

```bash
python scripts/build_cache.py --cdf-dir /path/to/ipix --out data/ipix_cache
python scripts/build_cache.py --cdf-dir /path/to/ipix --out data/ipix_cache --verify-only
```

`--verify-only` checks the 16 names against `metadata/target_cells.json` and prints
`16/16 cubes present`, or the missing list. It does not hash the cubes, because the archive is
written with the local zlib and the container bytes are not portable between NumPy versions.

The cache is used through one environment variable:

```bash
set PAPER1_CACHE=D:\data\ipix_cache          # Windows
export PAPER1_CACHE=/data/ipix_cache         # POSIX (not exercised in this release)
```

## 4. What to do if you cannot obtain the data

You can still verify every published number and both figures, because the released tables are
complete for that purpose:

```bash
python scripts/reproduce_tables.py
python scripts/reproduce_figures.py --check
```

Level 2 and Level 3 are unavailable without the recordings, and the validation report says so rather
than substituting synthetic data for IPIX measurements.

## 5. Rights reminder

Read `DATA_RIGHTS_AND_PROVENANCE.md` before copying anything into the repository. The `.gitignore`
blocks `data/ipix_cache/` and `*.cdf`, but a `git add -f` overrides it: check `git status` before any
commit that touches `data/`.
