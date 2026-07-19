# Quickstart (CPU, no downloads, ~3 minutes)

The quickstart trains a real distillation pipeline on a synthetic dataset
generated in memory. It demonstrates the mechanics end-to-end; real
distillation quality lives in the GPU demos (see Reproduce the paper).

```bash
embedkd fit --config configs/quickstart_cpu.yaml
```

You get a run directory:

```
runs/<timestamp>_quickstart/
├── fingerprint.yaml   # resolved config + seed + versions + git commit
├── log.jsonl          # one JSON record per epoch
├── best.pth           # best checkpoint by validation mAP
└── last.pth
```

Then walk the rest of the lifecycle:

```bash
embedkd diagnose --config configs/quickstart_cpu.yaml
embedkd eval     --config configs/quickstart_cpu.yaml --checkpoint runs/<id>/best.pth
embedkd deploy   --config configs/quickstart_cpu.yaml --checkpoint runs/<id>/best.pth
```

Every option in the config is explained in the Configuration reference.
Overrides never require editing files:

```bash
embedkd fit --config configs/quickstart_cpu.yaml --set train.epochs=5 --set train.seed=7
```
