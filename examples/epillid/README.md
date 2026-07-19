# Example: pill recognition with ePillID (demo D4)

This example shows EmbedKD on a real application domain via the generic
`csv_manifest` adapter. It lives in `examples/` on purpose: the core package
is domain-agnostic, and this directory is the template for bringing ANY
external dataset to the toolkit.

## Data (manual download, separate license)

ePillID (Usuyama et al., CVPR 2020 workshops) must be downloaded from its
official repository; review its license before use:
https://github.com/usuyama/ePillID-benchmark

Expected input: the benchmark's `all_labels.csv` (or `folds/*.csv`) with at
least the columns `image_path`, `label` (pill type id) and `is_ref`
(reference vs consumer photo).

## Build the manifest

```bash
python examples/epillid/prepare_manifest.py \
    /path/to/ePillID_data/all_labels.csv \
    /path/to/ePillID_data/classification_data \
    --out examples/epillid/manifest.csv
embedkd datasets validate csv_manifest:/path/to/ePillID_data/classification_data \
    --set data.manifest=examples/epillid/manifest.csv
```

The script maps the benchmark onto EmbedKD's retrieval protocol:

| ePillID concept | Manifest column |
|---|---|
| training fold classes | `split=train` |
| held-out classes, reference images | `split=gallery` |
| held-out classes, consumer images | `split=query` |

Gallery = reference photos and query = consumer photos is exactly the
real-world deployment scenario (match a phone photo against the reference
database), which makes this the cross-domain case study of the paper.

## Train and evaluate

```bash
embedkd fit --config examples/epillid/config.yaml \
    --set data.root=/path/to/ePillID_data/classification_data
```
