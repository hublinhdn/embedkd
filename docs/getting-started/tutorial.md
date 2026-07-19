# Tutorial: distill on your own dataset

This walkthrough takes a brand-new dataset from folder to distilled,
evaluated, deployable student. Every command below runs on a laptop CPU in
minutes; swap in your real data and a GPU for real work. (The test suite
executes this exact flow, so the tutorial cannot drift from the code.)

## 1. Point EmbedKD at your images

The zero-code path is one folder per class:

```
data/tutorial/
├── class_00/  img_000.jpg ...
├── class_01/  ...
```

No dataset of your own yet? Generate a toy one and follow along:

```python
from pathlib import Path
from PIL import Image

for c in range(6):
    d = Path(f"data/tutorial/class_{c:02d}")
    d.mkdir(parents=True, exist_ok=True)
    for i in range(12):
        Image.new("RGB", (80, 80),
                  (40 * c % 255, 30 * i % 255, (60 + 25 * c) % 255)
                  ).save(d / f"img_{i:03d}.jpg")
```

Data with its own structure fits through the `csv_manifest` adapter instead;
see [Bring your own dataset](../guides/bring-your-own-dataset.md).

## 2. Health-check before anything else

```bash
embedkd datasets validate image_folder:data/tutorial
```

```
   num_classes: 6
     train: images=36, classes=6, min=6, median=6, max=6
   gallery: images=18, ...
     query: images=18, ...
Result: OK
```

This catches unreadable images, empty splits, and classes thinner than the
sampler's K before you spend any training time. It also generates the
train/gallery/query split and freezes it to
`data/tutorial/embedkd_generated_split.csv`; every later run reuses that
file, so your split never silently re-randomises.

## 3. Write a config (about 20 lines)

Save as `tutorial.yaml`:

```yaml
run: {tag: tutorial}

teacher:
  backbone: resnet50
  weights: random          # see the note below for real teachers
  embed_dim: 128
student:
  backbone: resnet18
  pretrained: false        # true for real work
  embed_dim: 128           # must match the teacher for cosine/mse

head:
  losses: {sce: 1.0, triplet: 1.0}
  sce: {label_smoothing: 0.1}

distill: {objective: cosine, alpha: 10.0}

data:
  adapter: image_folder
  root: data/tutorial
  input_size: 64           # 224 for real work
  sampler: pk
  p_classes: 4
  k_samples: 4
  num_workers: 0

train: {epochs: 2, amp: false, seed: 42, warmup_epochs: 0}
eval: {batch_size: 64}
```

Everything not written here takes a documented default; the full annotated
schema is in the [Configuration reference](../guides/configuration.md).

**Where does a real teacher come from?** Three options for
`teacher.weights`: a checkpoint you already have; `pretrained` (ImageNet,
for quick trials); or train one with EmbedKD itself by running this same
config with `--set student.backbone=resnet50 --set distill.alpha=0`
(standalone mode) and pointing `teacher.weights` at its `best.pth`.

## 4. Ask whether distillation is worth it (before training)

```bash
embedkd diagnose --config tutorial.yaml
```

The report gives teacher-student CKA, backbone capacity ratio, and a risk
level. Reading guide from our published study: the strongest predictor of
distillation gain is the teacher-student performance gap, so a teacher that
barely beats a standalone student will transfer little; CKA and capacity add
context. A nonsense pairing (teacher smaller than the student) shows up here
in seconds as a capacity ratio below 1.

## 5. Train

```bash
embedkd fit --config tutorial.yaml
```

The run directory is the contract:

```
runs/<timestamp>_tutorial/
├── fingerprint.yaml   # resolved config + seed + versions + git commit
├── log.jsonl          # one JSON record per epoch, per-loss components included
├── best.pth           # best checkpoint by validation mAP
└── last.pth
```

## 6. Evaluate

```bash
embedkd eval --config tutorial.yaml --checkpoint runs/<id>/best.pth
```

You get image-level `map` (all relevant gallery images must rank high),
`mrr` and `r1`/`r5` (how early the first correct hit appears), and
`retention`: student mAP as a fraction of the teacher's, the single most
telling distillation number.

## 7. Iterate on the pair and the losses, config-only

Swap architectures without editing any file:

```bash
embedkd diagnose --config tutorial.yaml \
  --set teacher.backbone=convnext_tiny --set student.backbone=efficientnet_b0
```

`embedkd backbones` lists the validated menu; anything else in timm runs
under `--set backbone_policy=experimental` with an honest warning.

Change the loss mix the same way:

```bash
embedkd fit --config tutorial.yaml \
  --set 'head.losses={sce: 1.0, contrastive: 0.5}' \
  --set 'distill.objective={cosine: 1.0, rkd: 0.5}'
```

`log.jsonl` then reports `task_sce`, `task_contrastive`, `distill_cosine`,
`distill_rkd` per epoch, so re-weighting is a decision made from data. The
`losses` dict replaces the default outright: what you write is exactly what
trains.

## 8. Ship it

```bash
embedkd extract --config tutorial.yaml --checkpoint runs/<id>/best.pth \
    --out embeddings.npy --save-labels        # feed FAISS or any vector index
embedkd deploy --config tutorial.yaml --checkpoint runs/<id>/best.pth
```

`deploy` refuses to hand you an ONNX file whose outputs do not match the
torch model, then reports size, CPU latency, and FPS.

## Going further

- The `configs/` directory holds the exact recipes behind the paper's
  published numbers (D1 to D4); they are the reference for real
  hyperparameters (224px, 60 epochs, AMP, two-tier learning rate).
- `embedkd reproduce d1_cub200 --eval-only` verifies our published results
  on your machine; see REPRODUCE.md.
- Writing your own objective, task loss, or dataset adapter takes about
  fifteen lines: see the Extend section, whose examples are executed by CI.
