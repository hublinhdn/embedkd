# Expected results

Machine-generated specs consumed by `embedkd reproduce <demo_id>`.

Each `<demo_id>.json` is written by `scripts/make_expected_results.py` from
finished training runs; values are never edited by hand. Fields:

- `config`: the config a reviewer re-runs.
- `checkpoint`: released student weights backing `reproduce --eval-only`.
- `expected`: metric -> {value, tolerance} for the primary row; tolerance is
  2x the std over seed replicates (floor 0.005).
- `all_rows`: every objective/variant of the demo, for the paper table.

Populated by the G3 experiment campaign (see REPRODUCE.md).
