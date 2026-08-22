#!/usr/bin/env bash
# Revision experiment E11: MSE at a distillation weight matched to cosine.
#
# Reviewer 3 asks whether the cosine and MSE comparison is fair. It is not,
# and the factor is exact. For L2-normalised embeddings ||s - t||^2 = 2 - 2cos,
# so with reduction='mean' over batch AND embedding dimension D:
#
#   mse_loss = (1/D) * mean_b(2 - 2cos) = (2/D) * cosine_loss
#
# With embed_dim 512 that is cosine_loss / 256. At the same nominal weight of
# 10 the MSE distillation term is therefore 256 times weaker, which is why the
# published MSE row lands exactly on the no-KD baseline. Matching the effective
# scale needs alpha = 10 * 256 = 2560.
#
# Three seeds, because reviewers 1, 2 and 3 all ask for variance on the
# objectives that carry a comparison.
set -euo pipefail

EMBEDKD=${EMBEDKD:-embedkd}
D1_TEACHER=${D1_TEACHER:-runs/d1_cub200_teacher/best.pth}
ALPHA=${ALPHA:-2560.0}

for seed in 42 43 44; do
    echo "### mse matched scale, alpha ${ALPHA}, seed ${seed}"
    "${EMBEDKD}" fit \
        --config configs/d1_cub200_mse.yaml \
        --set teacher.weights="${D1_TEACHER}" \
        --set distill.alpha="${ALPHA}" \
        --set train.seed="${seed}" \
        --set run.tag="e11_d1_mse_matched_s${seed}"
done
