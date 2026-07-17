# Teacher-student diagnostics

The question no generic KD framework asks: is this pair worth distilling at
all? Distillation can fail in a specific, sneaky way: the student's geometry
aligns with the teacher (CKA goes up) while retrieval quality goes down.

## Before training

```bash
embedkd diagnose --config my.yaml --out report.json
```

Reports:

- `cka_pre`: linear CKA (Kornblith et al., 2019) between teacher and student
  embeddings on a probe set;
- `capacity_ratio`: parameter ratio;
- `risk`: LOW / MODERATE / HIGH.

## Thresholds

The risk levels use `RISK_THRESHOLDS = {"high": 0.35, "moderate": 0.60}` on
pre-distillation embedding CKA. These are heuristics derived from the
negative-transfer analysis in the authors' fine-grained recognition study
(see the paper's references); they are constants in
`embedkd.diagnostics.cka`, documented here so nobody mistakes them for laws
of nature. Treat MODERATE/HIGH as "run a short pilot before committing
GPU-days", not as a verdict.

## After training

`distill_report(...)` classifies the outcome:

| Pattern | Meaning |
|---|---|
| `improved` | metric went up; distillation helped |
| `aligned_but_worse` | CKA up, metric down; the student imitated geometry at the cost of the task |
| `diverged` | CKA and metric both down |

`embedkd.diagnostics.plots` renders both reports as figures
(`pip install "embedkd[plots]"`).
