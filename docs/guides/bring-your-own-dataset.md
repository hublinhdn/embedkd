# Bring your own dataset

Three paths, from zero code to twenty lines.

## Path 1: image folder (zero code)

```
data/my/
├── class_a/  img001.jpg ...
├── class_b/  ...
```

```yaml
data:
  adapter: image_folder
  root: data/my
  split: {mode: auto, gallery_ratio: 0.5}
```

The auto split is class-stratified, written once to
`embedkd_generated_split.csv` inside the data root, and re-read on every
subsequent run: your split is frozen the first time it materialises.

## Path 2: CSV manifest (structured data)

```yaml
data:
  adapter: csv_manifest
  root: /data/images
  manifest: my_manifest.csv
```

Column contract:

| Column | Required | Meaning |
|---|---|---|
| `path` | yes | relative to `data.root`, or absolute |
| `label` | yes | string or int; mapped to indices internally |
| `split` | no | `train` / `gallery` / `query`; omit to auto-split |
| `domain` | no | reserved for cross-domain setups |

See `examples/epillid/` for a complete real-world manifest builder.

## Health check first

```bash
embedkd datasets validate csv_manifest:/data/images --set data.manifest=my_manifest.csv
```

Catches missing files, unreadable images, empty splits, and classes with
fewer than K images (which the PK sampler would silently oversample). Run it
before every first training on new data; it is much cheaper than discovering
the problem at epoch 30.

## Path 3: custom adapter (about 20 lines)

Register a dataset adapter when your data cannot be expressed as a folder or
a CSV; see [Custom dataset adapter](../extend/custom-dataset-adapter.md).
The contract: `build(data_cfg, train_tf, eval_tf) -> DataBundle` where the
bundle carries `train`, `gallery`, `query` torch datasets plus `num_classes`
and `train_labels`. Adapters never choose transforms; the engine passes them
in so evaluation stays protocol-consistent.
