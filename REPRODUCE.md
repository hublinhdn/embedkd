# Reproducing the published results

Every number in the EmbedKD paper is backed by a config, a seed, a one-line
command, and a tolerance-checked expected value. The `embedkd reproduce`
command re-runs a demo and grades itself (exit code 2 on mismatch).

> Status: the expected-results tables below are produced by the G3 experiment
> campaign and land here together with `expected_results/*.json`. Until then,
> `embedkd reproduce --list` reports which demos are available.

## Quick verification (CPU, minutes)

Published student checkpoints are attached to GitHub Releases; with them the
whole table can be verified without a GPU:

```bash
embedkd reproduce d1_cub200 --eval-only
```

## Full reruns (GPU)

| Demo | Command | Reference hardware | Wall clock |
|---|---|---|---|
| D1 CUB-200, 4 objectives | `bash scripts/run_d1_cub200.sh` | 1x consumer GPU | ~1-2 days |
| D2 Cars196 cross-family | `embedkd fit --config configs/d2_cars196_convnext_mobilenet.yaml` | same | ~0.5 day |
| D3 SOP scale test | `embedkd fit --config configs/d3_sop_cosine.yaml` | same | ~1 day |
| D4 ePillID case study | see `examples/epillid/README.md` | same | ~0.5 day |
| D5 diagnostics demo | produced by step 1 of `run_d1_cub200.sh` | - | minutes |

## Determinism contract

1. Same machine, same seed: bit-exact metrics (covered by an automated test).
2. Different GPU or driver: expect deviations within the published tolerances,
   which are 2x the standard deviation over seeds {42, 43, 44} of the primary
   run, with a floor of 0.005.
3. Every run writes `fingerprint.yaml` (resolved config, seed, library
   versions, git commit, hardware); attach it when reporting discrepancies.

## How the expected values are generated

`scripts/make_expected_results.py` reads the best evaluated epoch from each
run's `log.jsonl` and writes `expected_results/<demo>.json`. No number is ever
typed by hand.
