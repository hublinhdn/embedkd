# Changelog

All notable changes to EmbedKD are documented here. The project follows
semantic versioning; the config schema is part of the public API.

## 0.1.4 - 2026-08-23

Revision round for the SoftwareX submission. No published metric changed; the
numbers of 0.1.3 reproduce bit-exactly, now verified by retraining rather than
by reloading a checkpoint.

Verification
- `parity_report` draws probes from real images, and reports maximum absolute
  and relative error next to the cosine agreement. The element-wise relative
  error is reported alongside one normalised by the vector norm, because the
  former is inflated by coordinates near zero.
- `retrieval_parity` recomputes mAP and Recall@k through the exported graph
  using the same evaluator, so a ranking change fails the check even when the
  embeddings agree closely.
- `export_onnx(dynamic_spatial=True)` frees the height and width axes. Off by
  default: accepting an input shape and being correct at it are separate
  properties. Verified at 160, 224 and 288 on all five validated backbones.
- `fetch_release_checkpoints.py` verifies the SHA-256 of what it downloads and
  refuses to continue on a mismatch.

Reproducibility record
- `requirements.lock` and `docs/reference-environment.md` are generated on the
  machine that produced the published results: operating system, kernel, GPU,
  driver, CUDA, cuDNN, all installed packages, checkpoint checksums, and which
  optional dependency each test file needs. Regenerating reproduces them byte
  for byte.
- Checkpoint checksums added to `expected_results/*.json`. Additive only.
- `server_setup.sh` installed torch from the cu121 index while the reference
  machine runs a cu130 build, so it never reproduced that environment. Fixed
  and pinned.
- `REPRODUCE.md` separates the evaluation level of reproduction from the
  training level, and maps each published run to the commit its fingerprint
  names. Those commits predate a pre-release history squash and are now
  published as the `archive/pre-squash-history` branch.
- CITATION.cff and the README badge use the Zenodo concept DOI, which always
  resolves to the newest version, instead of a per-version DOI that goes stale.

Tooling
- The dev extra bounds ruff to the 0.16 line and the lint rule set is declared
  in `pyproject.toml`. CI had been failing on unchanged code because a new ruff
  release widened its defaults.

Figures and scripts
- `make_qualitative_figure.py` defaults to a selection that alternates between
  queries distillation corrects and queries it degrades, and always prints both
  counts. Generated figures default into the ignored `figures/` directory.
- `scripts/revision/` holds the experiments run for the revision, each one
  producing the numbers quoted in the paper.

## 0.1.3 - 2026-07-20

Metadata and hygiene fixes surfaced by an independent review; no published
metric changed.

- Package version matches the release again (pyproject and `__init__` were
  stuck at 0.1.0 while everything else said 0.1.2).
- Repository and homepage URLs corrected to `github.com/hublinhdn/embedkd`
  (previously pointed at a non-existent org); removed a dead documentation URL.
- README quickstart is clone-first: `configs/` ship with the repo, not the wheel.
- `verify_teacher_metrics.py` skips a demo cleanly (missing dataset, manifest,
  or images root) instead of crashing; run `... d4_epillid --set data.root=...`
  for ePillID.
- `make_expected_results.py` preserves the frozen teacher block on regeneration;
  REPRODUCE.md and expected_results/README.md corrected to state that student
  rows are machine-generated while teacher metrics are frozen from the released
  checkpoints and re-verified by `verify_teacher_metrics.py`.

## 0.1.2 - 2026-07-20

Reproducibility hardening. Same checkpoints and published results as 0.1.0/0.1.1;
no metric changed.

- Teacher retrieval metrics (map, r1) frozen into `expected_results/*.json`
  and re-checkable with `scripts/verify_teacher_metrics.py`. Previously the
  teacher numbers were recorded by hand and unverified; a wrong D2 teacher mAP
  was caught this way and corrected (0.476 was a stale value; the released
  checkpoint gives 0.525).
- REPRODUCE.md documents teacher verification and the `data.root` argument the
  ePillID (D4) demo needs (its manifest stores paths relative to the images root).

## 0.1.1 - 2026-07-20

Additive only. Same checkpoints and expected results as 0.1.0; the published
D1-D5 numbers are unaffected and `reproduce` still passes.

- Per-component loss values logged each epoch (`task_sce`, `distill_rkd`,
  ...) alongside the aggregate task/distill losses.
- New-user tutorial (`docs/getting-started/tutorial.md`), executed by the
  test suite.
- Configuration reference synced with the full current schema
  (lr_backbone, grad_clip, relational_ramp, logit_scale, output_stride).
- `scripts/make_qualitative_figure.py`: qualitative retrieval figure
  (query vs top-k neighbours, before/after distillation).

## 0.1.0 - 2026-07-19

First release. Ships the complete D1-D5 reproduction suite
(`expected_results/`, REPRODUCE.md) with released checkpoints; verified
end-to-end with `embedkd reproduce d1_cub200 --eval-only` (PASS, bit-exact
on the reference machine).

- Distillation objectives: cosine, mse, kl, rkd (fp32-safe), weighted combos,
  registry for user objectives.
- Task losses: sce, arcface, triplet (batch-hard), contrastive.
- Backbone policy: 5 validated backbones, experimental opt-in for other timm
  models.
- Data: image_folder and csv_manifest adapters with frozen auto-splits,
  built-in CUB-200-2011 / Cars196 / Stanford Online Products adapters,
  PK sampler, dataset health check (datasets validate).
- Engine: deterministic training loop, AMP with fp32 loss island,
  run fingerprints, JSONL logs.
- Evaluation: gallery-query mAP / R@k, cross-domain, teacher retention.
- Diagnostics: linear CKA, pre-distillation compatibility report,
  post-distillation outcome classification, plots.
- Deploy: ONNX export with mandatory parity check, CPU latency benchmark.
- CLI: fit, eval, diagnose, extract, deploy, reproduce, backbones, datasets.
