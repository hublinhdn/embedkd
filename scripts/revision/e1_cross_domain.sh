#!/usr/bin/env bash
# Revision experiment E1: does distillation help or hurt cross-domain transfer?
#
# Evaluation only. Every checkpoint already exists, so nothing is trained here.
#
#   D1 students (ResNet18, trained on CUB-200)     evaluated on Cars196
#   D2 students (MobileNetV3, trained on Cars196)  evaluated on CUB-200
#
# For each source domain both the no-KD baseline and the distilled student are
# evaluated under identical conditions, so the result answers directly whether
# distillation improves or degrades transfer.
#
# Usage:
#   bash scripts/revision/e1_cross_domain.sh
# Every path can be overridden through the environment, e.g. OUT_DIR=... .

set -euo pipefail

EMBEDKD=${EMBEDKD:-embedkd}
OUT_DIR=${OUT_DIR:-runs/revision/e1_cross_domain}

D1_CONFIG=${D1_CONFIG:-configs/d1_cub200_cosine.yaml}
D2_CONFIG=${D2_CONFIG:-configs/d2_cars196_convnext_mobilenet.yaml}

D1_TEACHER=${D1_TEACHER:-runs/d1_cub200_teacher/best.pth}
D2_TEACHER=${D2_TEACHER:-runs/d2_cars196_teacher/best.pth}

D1_DISTILLED=${D1_DISTILLED:-runs/20260717_164702_d1_cosine_s42/best.pth}
D1_NO_KD=${D1_NO_KD:-runs/20260718_015853_d1_student_baseline/best.pth}
D2_DISTILLED=${D2_DISTILLED:-runs/20260718_084411_d2_cosine_s42/best.pth}
D2_NO_KD=${D2_NO_KD:-runs/20260718_091247_d2_no_kd/best.pth}

mkdir -p "$OUT_DIR"

run_eval () {
    local name=$1 config=$2 teacher=$3 ckpt=$4 target=$5
    echo "### ${name}: $(dirname "${ckpt}") evaluated on ${target}"
    "${EMBEDKD}" eval \
        --config "${config}" \
        --set teacher.weights="${teacher}" \
        --set data.protocol=cross_domain \
        --set data.target.adapter="${target}" \
        --checkpoint "${ckpt}" \
        --target \
        | tee "${OUT_DIR}/${name}.json"
    echo
}

run_eval cub_to_cars_distilled "${D1_CONFIG}" "${D1_TEACHER}" "${D1_DISTILLED}" cars196
run_eval cub_to_cars_no_kd     "${D1_CONFIG}" "${D1_TEACHER}" "${D1_NO_KD}"     cars196
run_eval cars_to_cub_distilled "${D2_CONFIG}" "${D2_TEACHER}" "${D2_DISTILLED}" cub200
run_eval cars_to_cub_no_kd     "${D2_CONFIG}" "${D2_TEACHER}" "${D2_NO_KD}"     cub200

echo "### done, results in ${OUT_DIR}"
