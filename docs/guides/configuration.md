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
  output_stride: null       # 16 enables the last-stride trick (resnet family)

student:
  backbone: resnet18
  pretrained: true          # imagenet initialisation
  embed_dim: 512            # must equal teacher's for cosine/mse objectives
  output_stride: null

head:
  pooling: gem              # gem | gap (transformers fall back to token mean)
  gem_p: 3.0
  gem_p_trainable: false
  normalize: true           # keep true for retrieval protocols
  logit_scale: 64.0         # cosine-classifier scale for sce / kl logits
  losses:                   # task losses and weights (registry names).
    sce: 1.0                # NOTE: this dict REPLACES the default entirely;
    triplet: 1.0            # write exactly the set of losses you want.
  sce: {label_smoothing: 0.1}
  triplet: {margin: 0.3, mining: batch_hard}
  arcface: {margin: 0.35, scale: 64.0}
  contrastive: {margin: 1.0}

distill:
  objective: cosine         # name, or weighted dict {cosine: 10.0, rkd: 1.0}
  alpha: 10.0               # 0 disables distillation (standalone training)
  kl: {temperature: 4.0}    # per-objective params live under the same name
  rkd: {distance_weight: 25.0, angle_weight: 50.0}
  relational_ramp:          # relational objectives (rkd) stay OFF until
    start_epoch: null       # start_epoch (null = after LR warmup), then
    epochs: 5               # ramp 0 -> 1 over this many epochs

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
  lr: 3.0e-4                # head / classifier / loss parameters
  lr_backbone: null         # null = lr / 10 (two-tier fine-tuning)
  weight_decay: 5.0e-2
  scheduler: cosine         # cosine | step | none
  warmup_epochs: 5
  amp: true                 # losses always run fp32 regardless
  grad_clip: 5.0            # global-norm clipping (0 disables)
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

## Semantics worth knowing

- **`head.losses` replaces, never merges.** `losses: {triplet: 1.0}` means
  triplet only; a default loss cannot leak in. Every other mapping deep-merges
  key by key.
- The training loop uses two learning rates: the pretrained backbone runs at
  `lr_backbone` (default `lr / 10`) and everything freshly initialised runs
  at `lr`. Single-rate fine-tuning collapses open-set retrieval.
- Classifier heads are created only when needed (`sce` in losses or `kl`
  objective); at SOP-scale class counts this saves millions of parameters.
- Per-component loss values are logged each epoch (`task_sce`,
  `distill_rkd`, ...), so re-weighting decisions can be made from
  `log.jsonl` instead of guesswork.
- `--set` accepts YAML-typed values, including dicts:
  `--set 'head.losses={sce: 1.0, contrastive: 0.5}'`.

The `configs/` directory of the repository contains the exact configurations
behind every published number (quickstart plus the D1 to D4 demos); they are
working examples of everything above.
