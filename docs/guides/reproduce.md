# Reproduce the paper

See REPRODUCE.md in the repository root for the authoritative table. Summary:

```bash
embedkd reproduce --list                 # what is available
embedkd reproduce d1_cub200 --eval-only  # verify on CPU with released weights
embedkd reproduce d1_cub200              # full retrain (GPU, ~1 day)
```

`reproduce` compares your metrics against `expected_results/<demo>.json`
within published tolerances and exits with code 2 on any mismatch, so it can
gate CI pipelines.

Determinism contract: same machine + same seed is bit-exact; across GPUs and
drivers expect deviations within the tolerances (2x std over seeds 42-44,
floor 0.005). Attach your run's `fingerprint.yaml` when reporting
discrepancies.
