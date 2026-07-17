# FAQ / Troubleshooting

**Training "runs" but mAP is frozen from some epoch on.**
Classic silent fp16 failure in geometric losses. EmbedKD computes all losses
in fp32 outside autocast precisely for this, so if you see it, check that
custom code is not re-enabling autocast around a loss.

**`Too many open files` (Errno 24) when creating many DataLoaders.**
Creating many loaders with `num_workers > 0` in one process leaks file
descriptors on some platforms. Reuse loaders, or lower `data.num_workers`,
or raise the ulimit.

**Cars196 download fails.**
The original Stanford links are frequently offline. Follow the manual layout
in the adapter's error message and convert annotations with
`scripts/convert_cars196.py`.

**`'X' is not a validated backbone`.**
Deliberate. Set `backbone_policy: experimental` to run any timm backbone
with a warning, or use `embedkd backbones` to see the validated list.

**My numbers differ slightly from the published table on a different GPU.**
Expected. See the determinism contract in REPRODUCE.md; deviations should
stay within the published tolerances. Same machine + same seed must be
bit-exact.

**KL objective errors about missing logits.**
`kl` needs classifier logits on both sides: include `sce` in `head.losses`
and use a teacher checkpoint trained with a classifier.

**Where did my split come from?**
Auto-splits are frozen to `embedkd_generated_split.csv` in the data root on
first use. Delete that file only if you intend to re-randomise the split.
