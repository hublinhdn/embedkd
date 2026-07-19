# Changelog

All notable changes to EmbedKD are documented here. The project follows
semantic versioning; the config schema is part of the public API.

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
