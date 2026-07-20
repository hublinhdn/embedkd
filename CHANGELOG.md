# Changelog

All notable changes to EmbedKD are documented here. The project follows
semantic versioning; the config schema is part of the public API.

## Unreleased

- `verify_teacher_metrics.py` skips a demo cleanly (with a one-line hint)
  when its images root is not provided, instead of crashing deep in the
  DataLoader; run `... d4_epillid --set data.root=<images root>` for ePillID.

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
