# Changelog

All notable changes to EmbedKD are documented here. The project follows
semantic versioning; the config schema is part of the public API.

## Unreleased (0.1.0.dev0)

First development snapshot.

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
