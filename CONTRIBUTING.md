# Contributing to EmbedKD

Thanks for your interest. A few ground rules keep the toolkit reliable.

## Development setup

```bash
git clone https://github.com/embedkd/embedkd && cd embedkd
pip install -e ".[dev]"
pytest -q          # the whole suite runs on CPU in seconds
ruff check .
```

## Adding a component (no core edits needed)

Distillation objectives, task losses and dataset adapters are registered via
decorators; see the `extend` section of the docs. If your component is useful
to others, open a PR that adds it plus a unit test with a hand-checkable
reference value (shape-only tests are not enough).

## Adding a backbone to the validated list

The supported-backbone list is a frozen, tested claim, not a wish list.
To promote a backbone: add one entry to `SUPPORTED_BACKBONES` (or
`EXPERIMENTAL_VERIFIED`) in `embedkd/models/backbones.py` in a PR. CI runs the
smoke test for every listed entry; green CI is the acceptance bar. Until it is
merged, any timm backbone already runs under `backbone_policy: experimental`.

## Non-negotiables

1. Every published number must be reproducible: config + seed + fingerprint.
2. Losses are computed in fp32 outside autocast; do not move them inside.
3. The core package stays domain-agnostic (an automated test enforces this).
4. Breaking config-schema changes need a deprecation note in the CHANGELOG.

## Reporting issues

Include the `fingerprint.yaml` of the affected run; it answers most questions
(versions, config, seed) before we have to ask.
