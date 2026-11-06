# Data

This directory holds the **IPIX complex-cube cache**, which is not distributed with this
repository. The recordings are third-party data and the cache is a derived copy of them.

## What is needed

Sixteen acquisitions, named in `../metadata/target_cells.json`, each as one
`<stem>_cubes.npz` holding the complex VV/VH cubes, the pulse repetition frequency and the range
axis. The full list, the excluded-target cells and the provenance of each acquisition are in that
file and in `../protocol/DATA_PARTITION_MANIFEST.csv`.

## Building the cache

```bash
python scripts/build_cache.py --cdf-dir /path/to/ipix --out data/ipix_cache
python scripts/build_cache.py --cdf-dir /path/to/ipix --out data/ipix_cache --verify-only
```

`--verify-only` reports `16/16 cubes present` or lists the missing acquisitions. A cache built
elsewhere can be used by setting `PAPER1_CACHE`.

## Rights

Do **not** commit cubes or recordings to a public repository. Read
`../docs/DATA_RIGHTS_AND_PROVENANCE.md` before moving any file, and
`../docs/DATA_DOWNLOAD_INSTRUCTIONS.md` for how to obtain the recordings legitimately. The paper's
data-availability statement is written to match this restriction.

## What the released package does without the data

Everything in Level 1: all six tables, both figures, and every published number, from
`../results/`. The cache is needed only to re-run detectors (Level 2) or training (Level 3).
