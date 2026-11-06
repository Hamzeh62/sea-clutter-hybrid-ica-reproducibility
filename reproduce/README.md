# One-command recipes

Run from the repository root. Windows paths shown; the commands are shell-neutral.

| goal | command | needs | time |
|---|---|---|---|
| is this checkout ready? | `python scripts/doctor.py` | nothing | 2 s |
| verify the released files are unmodified | `python scripts/verify_artifacts.py` | nothing | 3 s |
| every published number | `python scripts/reproduce_tables.py` | nothing | 3 s |
| both figures | `python scripts/reproduce_figures.py --check` | matplotlib, Pillow | 10 s |
| all tests | `python -m pytest -q` | pytest; cache for 3 tests | 5 s |
| synthetic smoke test | `python scripts/smoke_test.py --checkpoint checkpoints/best_s0.pt` | torch, weights | 20 s |
| verify the weights | `python scripts/checkpoint_fixture.py --verify --checkpoints checkpoints` | torch, weights | 10 s |
| rebuild the cube cache | `python scripts/build_cache.py --cdf-dir /path/to/ipix --out data/ipix_cache` | IPIX recordings | 5–10 min |
| re-run one released job | `python scripts/validate_against_released_scores.py --released <results>/runs/minimal_v1_20261029 --job "h1|fold1|hybrid|c0"` | cubes, weights, score archive | 2–3 min |

## Environment variables

| variable | default | purpose |
|---|---|---|
| `PAPER1_CACHE` | `data/ipix_cache` | complex-cube cache |
| `PAPER1_CHECKPOINTS` | `checkpoints` | released `.pt` files |
| `PAPER1_OUTPUTS` | `outputs` | everything the pipeline writes |

## Recommended order for a reviewer

1. `verify_artifacts.py` — the files are the files.
2. `reproduce_tables.py` — the numbers in the paper.
3. `reproduce_figures.py --check` — the figures.
4. `pytest -q` — the protocol regressions.
5. If data and weights are available: `checkpoint_fixture.py --verify`, then
   `validate_against_released_scores.py`.

Each step is independent; a failure in step 5 does not invalidate steps 1–4, and the validation
report states which levels were exercised for this release.
