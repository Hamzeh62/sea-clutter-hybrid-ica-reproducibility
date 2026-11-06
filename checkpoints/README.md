# Checkpoints

Two of the nine trained weight files are required to reproduce the paper. The other seven are
development snapshots and are archived for provenance but are not used by any released result.

| file | role | size | SHA-256 |
|---|---|---|---|
| `best_s0.pt` | hybrid detector: network (M1) -> mixing head (M2) -> anchored complex ICA (M3) -> separated NMF (M4) | 3,893,891 B | `a0a49db9c24814706fa89b38c8fc1ee8b306fc15cf6a611f259148d18a06ce53` |
| `best_s0_nophys.pt` | physics-free ablation (same architecture, physics term removed from the loss) | 3,894,669 B | `2bcf10417a97f4d234853c97d43106331b33c0c0ab99f3e5b30ebdcd18411da0` |

Both were trained on **synthetic compound-Gaussian clutter**, not on IPIX recordings, so they leak no
evaluation data. They are 3.9 MB each and are therefore not committed to Git; they are distributed
in the archival record (`PAPER1_CHECKPOINTS_v1.0.0.zip`).

## Obtaining them

1. Download the checkpoint archive from the Zenodo record listed in `../docs/RELEASE_NOTES.md`
   (DOI assigned when the record is published; no DOI is invented here).
2. Extract `best_s0.pt` and `best_s0_nophys.pt` into this directory.
3. Verify:

```bash
python scripts/checkpoint_fixture.py --verify --checkpoints checkpoints
```

Expected: `CHECKPOINT FIXTURE VERIFIED`. The script checks the file hash, the parameter digest of the
`state_dict` (format-independent), and a deterministic forward pass of both networks on a fixed
synthetic CPI.

A different path can be used via `PAPER1_CHECKPOINTS` or `--checkpoints`.

## Loading safely

The files are PyTorch zip archives serialised with `torch.save`, which uses pickle. Load them only
from this archive, with `weights_only=True` where your PyTorch supports it:

```python
import torch
sd = torch.load("checkpoints/best_s0.pt", map_location="cpu", weights_only=True)
```

Do not load a checkpoint from an untrusted source. The fixture compares the parameter digest, so a
substituted file is detected even if the container bytes are re-serialised.

## What is *not* claimed

Training was not re-executed for this release. The checkpoints are the original bytes; their
reproduction from scratch is Level 3 in the README and is code-only, not validated.


**License:** Both author-owned trained checkpoints are published under CC BY 4.0; see `../DATA_LICENSE.md`. Original IPIX data are not included.
