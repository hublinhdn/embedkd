# Expected results

Machine-generated specs consumed by `embedkd reproduce <demo_id>`.

Each `<demo_id>.json` carries machine-generated student rows (written by
`scripts/make_expected_results.py` from finished training runs) plus a teacher
block frozen from the released checkpoints. Fields:

- `config`: the config a reviewer re-runs.
- `checkpoint`: released student weights backing `reproduce --eval-only`.
- `expected`: metric -> {value, tolerance} for the primary row; tolerance is
  2x the std over seed replicates (floor 0.005), or a flat 0.02 when fewer than
  two seed runs exist (D3, D4).
- `teacher`: `{backbone, map, r1}` of the teacher, frozen from the released
  checkpoint and re-verified by `scripts/verify_teacher_metrics.py`.
- `all_rows`: every objective/variant of the demo, for the paper table.

Populated by the G3 experiment campaign (see REPRODUCE.md).
