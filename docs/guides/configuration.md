# Configuration reference

One YAML file per run; defaults + your file + `--set` overrides are merged,
validated fail-fast, and the resolved result is dumped verbatim into the run's
`fingerprint.yaml`. What you configure is exactly what executes.

```yaml
run:
  tag: my_experiment        # names the run directory; default: config filename
  output_dir: runs

backbone_policy: supported  # supported | experimental (see `embedkd backbones`)

teacher:
  backbone: resnet50        # timm id from the validated list
  weights: teacher.pth      # path | pretrained (imagenet) | random
  embed_dim: 512

student:
  backbone: resnet18
  pretrained: true          # imagenet initialisation
  embed_dim: 512            # must equal teacher's for cosine/mse objectives

head:
  pooling: gem              # gem | gap (transformers fall back to token mean)
  gem_p: 3.0
  gem_p_trainable: false
  normalize: true           # keep true for retrieval protocols
  losses:                   # task losses and weights (registry names)
    sce: 1.0
    triplet: 1.0
  triplet: {margin: 0.2, mining: batch_hard}
  arcface: {margin: 0.5, scale: 30.0}

distill:
  objective: cosine         # name, or weighted dict {cosine: 10.0, rkd: 1.0}
  alpha: 10.0               # 0 disables distillation (standalone training)
  kl: {temperature: 4.0}    # per-objective params live under the same name
  rkd: {distance_weight: 1.0, angle_weight: 2.0}

data:
  adapter: image_folder     # image_folder | csv_manifest | cub200 | cars196 | sop | synthetic
  root: data/my
  manifest: null            # csv_manifest only
  input_size: 224
  protocol: gallery_query   # gallery_query | cross_domain
  split: {mode: auto, gallery_ratio: 0.5}
  sampler: pk               # pk | random (pk required with triplet/contrastive)
  p_classes: 16             # PK: P classes per batch
  k_samples: 4              # PK: K images per class (batch = P*K)
  num_workers: 4
  target: null              # cross_domain only: {adapter, root, manifest}

train:
  epochs: 60
  batch_size: 64            # ignored when sampler is pk
  optimizer: adamw          # adamw | sgd
  lr: 3.0e-4
  weight_decay: 1.0e-4
  scheduler: cosine         # cosine | step | none
  warmup_epochs: 1
  amp: true                 # losses always run fp32 regardless
  seed: 42
  eval_every: 1             # evaluate every N epochs (best tracked then)
  early_stopping: null      # or {metric: map, patience: 10}
  save_every: 0

eval:
  batch_size: 256
  metrics: [map, r1, r5]
  report_retention: true    # also report student/teacher mAP ratio
```

## Validation rules (fail at parse time, not at epoch 30)

1. `cosine`/`mse` objectives require equal teacher and student `embed_dim`.
2. `triplet`/`contrastive` task losses require `sampler: pk`.
3. Backbones outside the validated list require `backbone_policy: experimental`.
4. `protocol: cross_domain` requires a `data.target` block.
5. Unknown keys are rejected with a closest-match suggestion (typo guard).

## Notes

- Classifier heads are created only when needed (`sce` in losses or `kl`
  objective); at SOP-scale class counts this saves millions of parameters.
- `--set` accepts YAML-typed values: `--set head.losses='{triplet: 1.0}'`.
