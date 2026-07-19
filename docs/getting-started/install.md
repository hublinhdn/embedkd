# Install

```bash
pip install embedkd                  # core: torch, timm, numpy, pyyaml, pillow
pip install "embedkd[onnx]"          # + deploy (onnx, onnxruntime)
pip install "embedkd[plots]"         # + diagnostic figures (matplotlib)
pip install "embedkd[dev]"           # + pytest, ruff (contributors)
```

Supported: Python 3.10 to 3.12, the two most recent torch minor versions,
timm >= 1.0. CI tests the {torch N, N-1} x {py 3.10, 3.12} matrix on CPU.

GPU wheels: install torch AND torchvision for your CUDA version first, from
the same index
(`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`),
then `pip install embedkd`. Mixing indexes for the two packages breaks
torchvision at import time ("operator torchvision::nms does not exist").

Verify the installation:

```bash
embedkd --version
pytest --pyargs embedkd  # or: pytest -q  from a source checkout
```
