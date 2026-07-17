# Deployment

```bash
pip install "embedkd[onnx]"
embedkd deploy --config my.yaml --checkpoint runs/<id>/best.pth --out-dir deploy_out
```

Three steps, in order:

1. **Export** to ONNX (dynamic batch axis; GeM pooling is export-safe by
   construction).
2. **Parity check**: minimum cosine similarity between torch and onnxruntime
   outputs over random probes must exceed 0.9999. A mismatch raises and the
   CLI exits with code 2; an export that does not reproduce the model is a
   failure, not a warning.
3. **Benchmark**: CPU latency mean and std over N=100 runs after 10 warmup
   runs, FPS, parameter count and file size.

Output is a single JSON report; pair it with `embedkd eval` numbers to build
accuracy-versus-latency tables for model selection.
