# Reproducing the published results

Every number below is backed by a config, a seed, a one-line command, and a
tolerance-checked expected value in `expected_results/`. The
`embedkd reproduce` command re-runs a demo and grades itself
(exit code 2 on mismatch).

## Published results (v0.1.2)

Retrieval mAP, open-set gallery-query protocol, seed 42; tolerances are
2x the standard deviation over seeds {42, 43, 44} where replicates exist
(D1, D2), otherwise a conservative 0.02.

| Demo | Dataset | Pair | Teacher | no-KD student | cosine KD | KD gain |
|---|---|---|---|---|---|---|
| d1_cub200 | CUB-200-2011 | ResNet50 -> ResNet18 | 0.3394 | 0.2844 | 0.2882 +/- 0.005 | +0.004 |
| d3_sop | Stanford Online Products | ResNet50 -> ResNet18 | 0.6126 | 0.5296 | 0.5484 +/- 0.02 | +0.019 |
| d4_epillid | ePillID (case study) | ResNet50 -> ResNet18 | 0.5687 | 0.4530 | 0.4876 +/- 0.02 | +0.035 |
| d2_cars196 | Cars196 (cross-family) | ConvNeXt-T -> MobileNetV3 | 0.5251 | 0.2289 | 0.2797 +/- 0.005 | +0.051 |

The gain is monotone in the teacher-student performance gap; the
pre-training compatibility reports for all four pairs are produced by
`embedkd diagnose` (see the paper and `docs/concepts/diagnostics.md`).
D1 additionally compares four objectives (cosine / mse / kl / rkd); see
`expected_results/d1_cub200.json` for every row.

## Quick verification (CPU, minutes, no GPU, no training)

```bash
pip install -e .
python scripts/fetch_release_checkpoints.py d1_cub200   # downloads the released student
embedkd datasets download cub200 --root data            # one-time dataset fetch
embedkd reproduce d1_cub200 --eval-only
```

Expected output ends with PASS on every metric. Same for `d2_cars196`
(Cars196 requires the manual download described by the adapter's error
message), `d3_sop`, and `d4_epillid` (manual download, see
`examples/epillid/README.md`).

## Full reruns (GPU)

| Demo | Command | Reference hardware | Wall clock |
|---|---|---|---|
| D1 (4 objectives + baseline + 3 seeds) | `bash scripts/run_d1_cub200.sh` | RTX 3080 10GB | ~2.5 h |
| D2 | teacher, cosine, no-KD runs of `configs/d2_cars196_convnext_mobilenet.yaml` | same | ~1 h |
| D3 | same pattern on `configs/d3_sop_cosine.yaml` (`train.eval_every=10` recommended) | same | ~4 h |
| D4 | same pattern on `examples/epillid/config.yaml` | same | ~1 h |

Teachers are trained with `--set distill.alpha=0` (standalone mode); released
teacher checkpoints can be fetched with
`python scripts/fetch_release_checkpoints.py <demo> --teacher` to skip that
step.

## Verifying the teacher metrics

The Teacher column above is re-checkable against the released checkpoints
(the values are frozen in the `teacher` block of each `expected_results/*.json`):

```bash
python scripts/verify_teacher_metrics.py                 # D1, D2, D3
python scripts/verify_teacher_metrics.py d4_epillid \
    --set data.root=<ePillID classification_data dir>    # D4 needs the images root
```

D4 (ePillID) uses the `csv_manifest` adapter whose manifest stores image paths
relative to the images root, so every D4 command (`reproduce`, `verify`, `eval`,
`fit`) needs `--set data.root=<dir containing fcn_mix_weight/>`.

## Determinism contract

1. Same machine, same seed: bit-exact metrics (verified: the v0.1.2
   `reproduce --eval-only` run matched the published D1 numbers to 7 decimal
   places on the reference machine).
2. Different GPU or driver: expect deviations within the published
   tolerances.
3. Every run writes `fingerprint.yaml` (resolved config, seed, library
   versions, git commit, hardware); attach it when reporting discrepancies.

## How the numbers are generated

`scripts/make_expected_results.py` reads the best evaluated epoch from each
run's `log.jsonl` and writes `expected_results/<demo>.json`. No number is
typed by hand.
