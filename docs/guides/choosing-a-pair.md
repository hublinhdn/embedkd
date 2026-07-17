# Choosing a teacher-student pair

Run diagnostics before burning GPU-days:

```bash
embedkd diagnose --config my.yaml --out report.json
```

Reading the report:

1. **capacity_ratio** far above ~10x often correlates with poor transfer;
   consider an intermediate-size student or a stronger student init.
2. **cka_pre** LOW risk: proceed. MODERATE/HIGH: run a short pilot first
   (`--set train.epochs=5`) and check the trend before the full run.
3. After training, compare with `distill_report`: if you land in
   `aligned_but_worse`, lowering `distill.alpha` or switching from `mse` to
   `cosine`/`rkd` changes what the student is forced to imitate.

Validated pairs (numbers in the paper): resnet50 -> resnet18 (same family),
convnext_tiny -> mobilenetv3_large_100 (cross-family). Any other timm pair
runs under `backbone_policy: experimental`.
