#!/usr/bin/env bash
# Revision experiment E12: an end-to-end TRAINING reproduction of D1.
#
# Reviewer 2 points out that the headline reproduction command re-evaluates a
# released checkpoint, which exercises the evaluator and the checkpoint loader
# but says nothing about training determinism. `embedkd reproduce` without
# --eval-only retrains the demo from scratch and compares the fresh metrics
# against the frozen expected values, which is the missing evidence.
#
# It also settles a second question: the published D1 numbers were produced
# before the config semantics fix (losses replace instead of merge), so this
# run shows whether the current code path still lands on the published values.
#
# Usage: bash scripts/revision/e12_end_to_end_retrain.sh
set -euo pipefail

EMBEDKD=${EMBEDKD:-embedkd}
D1_TEACHER=${D1_TEACHER:-runs/d1_cub200_teacher/best.pth}
TAG=${TAG:-e12_retrain_d1_cosine_s42}

"${EMBEDKD}" reproduce d1_cub200 \
    --set teacher.weights="${D1_TEACHER}" \
    --set run.tag="${TAG}"
