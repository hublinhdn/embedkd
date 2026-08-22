#!/usr/bin/env bash
# Revision experiment E8: seed replicates for the objectives that the paper
# compares but never repeated.
#
# Reviewer 3 asks for seed level variability "for important comparisons such
# as RKD and MSE where possible", and states that a large scale repeated study
# is not required for a software demonstration. Reviewers 1 and 2 make the
# same point more broadly. Seeds 43 and 44 join the existing seed 42 run, so
# each objective ends with three.
set -euo pipefail

EMBEDKD=${EMBEDKD:-embedkd}
D1_TEACHER=${D1_TEACHER:-runs/d1_cub200_teacher/best.pth}

for objective in rkd mse kl; do
    for seed in 43 44; do
        echo "### ${objective}, seed ${seed}"
        "${EMBEDKD}" fit \
            --config "configs/d1_cub200_${objective}.yaml" \
            --set teacher.weights="${D1_TEACHER}" \
            --set train.seed="${seed}" \
            --set run.tag="e8_d1_${objective}_s${seed}"
    done
done
