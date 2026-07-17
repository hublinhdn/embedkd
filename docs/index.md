# EmbedKD

EmbedKD is a reproducibility-first toolkit for distilling metric embeddings:
it tells you whether a teacher-student pair is worth distilling, distills it,
evaluates it with retrieval protocols, and benchmarks the deployed result.

```bash
pip install embedkd
embedkd fit --config configs/quickstart_cpu.yaml   # CPU, ~3 minutes, no downloads
embedkd diagnose --config configs/quickstart_cpu.yaml
```

## The workflow

| Verb | Question it answers |
|---|---|
| `embedkd diagnose` | Is this teacher-student pair worth distilling at all? |
| `embedkd fit` | Train the student with joint metric learning + distillation. |
| `embedkd eval` | What are mAP / R@k, and how much of the teacher is retained? |
| `embedkd extract` | Give me the embeddings for my own index (FAISS, ...). |
| `embedkd deploy` | Does the ONNX export match, and how fast is it on CPU? |
| `embedkd reproduce` | Do I get the published numbers on my machine? |

`embedkd backbones` and `embedkd datasets` list what is supported.

## What EmbedKD is not

Not a general metric-learning library, not a classification KD framework,
not a distributed-training system. v0.1 is single-GPU, images only. Those
boundaries are deliberate; see the paper's roadmap for what comes next.
