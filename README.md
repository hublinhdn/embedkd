# EmbedKD

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21459157.svg)](https://doi.org/10.5281/zenodo.21459157)

EmbedKD is a reproducibility-first toolkit for distilling metric embeddings:
it tells you whether a teacher-student pair is worth distilling, distills it,
evaluates it with retrieval protocols, and benchmarks the deployed result.

> Status: v0.1.2 released, archived at [doi:10.5281/zenodo.21459157](https://doi.org/10.5281/zenodo.21459157).

## Why EmbedKD

Existing knowledge-distillation frameworks target classification. EmbedKD is
built for image retrieval and fine-grained recognition, where the product of
training is an embedding space, not logits:

1. Distillation directly in the normalised embedding space
   (cosine / MSE / KL / RKD), jointly with metric-learning task losses.
2. Retrieval evaluation protocol: gallery-query mAP and Recall@k, plus
   cross-domain zero-shot evaluation.
3. Teacher-student compatibility diagnostics (CKA-based) that run BEFORE you
   spend GPU-days, and classify the outcome afterwards.
4. Deployment benchmark: ONNX export with a mandatory numerical parity check,
   CPU latency, model size.

What EmbedKD is NOT: a general metric-learning library, a classification KD
framework, or a distributed-training system. v0.1 is single-GPU, images only.

## Quickstart (CPU, no downloads, ~3 minutes)

```bash
pip install embedkd
embedkd fit --config configs/quickstart_cpu.yaml
embedkd diagnose --config configs/quickstart_cpu.yaml
```

## The workflow

```bash
embedkd diagnose  --config c.yaml                  # worth distilling at all?
embedkd fit       --config c.yaml                  # train the student
embedkd eval      --config c.yaml --checkpoint p   # mAP / R@k (+ cross-domain)
embedkd extract   --config c.yaml --checkpoint p --out emb.npy
embedkd deploy    --config c.yaml --checkpoint p   # ONNX + parity + latency
embedkd reproduce d1_cub200 --eval-only            # verify published numbers
```

`embedkd backbones` lists the validated backbones; any other timm model runs
under `backbone_policy: experimental` with an explicit warning.

## Extending

```python
from embedkd import registry
from embedkd.objectives import DistillObjective

@registry.distill_objective("my_loss")
class MyLoss(DistillObjective):
    def forward(self, s_emb, t_emb, **kw):
        ...
```

Task losses and dataset adapters are registered the same way; see the docs.

## Reproducibility

Every run writes a fingerprint (resolved config, seed, library versions, git
commit) next to its logs. Published numbers come with one-line commands and
tolerance-checked expected results (`embedkd reproduce`). Same machine + same
seed is bit-exact; across GPUs, expect differences within the tolerances
documented in REPRODUCE.md.

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```

## License

MIT. See `LICENSE`.
