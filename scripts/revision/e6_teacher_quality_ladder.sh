#!/usr/bin/env bash
# Revision experiment E6 (demo D6): vary teacher quality, hold everything else.
#
# All three reviewers reject the gap to gain claim on the same two grounds:
# only four observations, and D2 changes dataset, teacher family, student
# family and capacity ratio at once; and both axes subtract the same no-KD
# score, so an arithmetic correlation can appear on its own.
#
# This experiment removes both objections. Dataset (CUB-200), student
# (ResNet18), objective (cosine at alpha 10), schedule and seed are fixed;
# only the teacher changes. Teachers of different quality are produced by
# training the same ResNet50 recipe for different numbers of epochs, each with
# its own cosine schedule, so every teacher is converged for its budget rather
# than being an arbitrary mid-training snapshot.
#
# Because the dataset and the student are fixed, the no-KD score is a CONSTANT
# across every point. A constant shared term cannot induce a correlation, which
# is precisely the objection raised in R1.2, R2.2 and R3.5.
#
# The 60 epoch teacher and its distilled student already exist as the published
# D1 pair, so only the shorter budgets are trained here.
set -euo pipefail

EMBEDKD=${EMBEDKD:-embedkd}
EPOCH_BUDGETS=${EPOCH_BUDGETS:-"5 15 30"}

for budget in ${EPOCH_BUDGETS}; do
    echo "### teacher ResNet50, ${budget} epochs"
    "${EMBEDKD}" fit \
        --config configs/d1_cub200_teacher.yaml \
        --set train.epochs="${budget}" \
        --set run.tag="e6_teacher_e${budget}"
done

for budget in ${EPOCH_BUDGETS}; do
    teacher=$(ls -dt runs/*_e6_teacher_e"${budget}" | head -1)/best.pth
    echo "### student ResNet18 distilled from the ${budget} epoch teacher (${teacher})"
    "${EMBEDKD}" fit \
        --config configs/d1_cub200_cosine.yaml \
        --set teacher.weights="${teacher}" \
        --set run.tag="e6_student_from_e${budget}"
done

echo "### done. The 60 epoch point is the published D1 pair."
