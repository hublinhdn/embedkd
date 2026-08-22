#!/usr/bin/env bash
# Revision experiment E15: a grid of teacher and student pairs with real outcomes.
#
# R1.1 and R2.1 ask for the diagnostic to be evaluated over "a substantially
# larger and more diverse collection of pairs" with their actual outcomes,
# rather than the four demonstrations. On CUB-200 a run costs about sixteen
# minutes, so a two teacher by three student grid is affordable.
#
# Teachers and students are the tier 1 backbones the paper already validates:
#   teachers  resnet50, convnext_tiny
#   students  resnet18, mobilenetv3_large_100, efficientnet_b0
#
# Every cell reports the compatibility report before training and the retrieval
# metrics after, so the diagnostic signals can be checked against outcomes that
# were actually measured rather than assumed.
#
# The resnet50 teacher, the resnet18 no-KD baseline and the resnet50 to
# resnet18 cosine cell already exist as the published D1 runs.
set -euo pipefail

EMBEDKD=${EMBEDKD:-embedkd}
TEACHERS=${TEACHERS:-"convnext_tiny"}
STUDENTS_NO_KD=${STUDENTS_NO_KD:-"mobilenetv3_large_100 efficientnet_b0"}

# 1. Missing teacher.
for backbone in ${TEACHERS}; do
    echo "### teacher ${backbone} on CUB-200"
    "${EMBEDKD}" fit \
        --config configs/d1_cub200_teacher.yaml \
        --set student.backbone="${backbone}" \
        --set run.tag="e15_teacher_${backbone}"
done

# 2. Missing no-KD baselines. distill.alpha 0 trains the student standalone.
for backbone in ${STUDENTS_NO_KD}; do
    echo "### no-KD baseline ${backbone} on CUB-200"
    "${EMBEDKD}" fit \
        --config configs/d1_cub200_teacher.yaml \
        --set student.backbone="${backbone}" \
        --set run.tag="e15_no_kd_${backbone}"
done

# 3. The distillation cells that do not exist yet.
resnet50_teacher=${RESNET50_TEACHER:-runs/d1_cub200_teacher/best.pth}
convnext_teacher=$(ls -dt runs/*_e15_teacher_convnext_tiny | head -1)/best.pth

distil () {
    local teacher_backbone=$1 teacher_ckpt=$2 student_backbone=$3
    echo "### ${teacher_backbone} distils ${student_backbone}"
    "${EMBEDKD}" fit \
        --config configs/d1_cub200_cosine.yaml \
        --set teacher.backbone="${teacher_backbone}" \
        --set teacher.weights="${teacher_ckpt}" \
        --set student.backbone="${student_backbone}" \
        --set run.tag="e15_${teacher_backbone}_to_${student_backbone}"
}

distil resnet50      "${resnet50_teacher}" mobilenetv3_large_100
distil resnet50      "${resnet50_teacher}" efficientnet_b0
distil convnext_tiny "${convnext_teacher}" resnet18
distil convnext_tiny "${convnext_teacher}" mobilenetv3_large_100
distil convnext_tiny "${convnext_teacher}" efficientnet_b0

echo "### done. resnet50 to resnet18 is the published D1 cell."
