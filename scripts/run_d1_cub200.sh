#!/usr/bin/env bash
# D1 reproduction pipeline on CUB-200-2011. Run inside tmux from the repo root:
#   bash scripts/run_d1_cub200.sh
# Steps: teacher -> 4 objectives (seed 42) -> cosine seeds 43/44 -> diagnose,
# deploy and summary for the cosine student. Each step resumes from existing
# runs, so the script is safe to re-run after an interruption.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

latest_run() {  # latest_run <tag> -> newest runs/<timestamp>_<tag> directory
    ls -dt runs/*_"$1" 2>/dev/null | head -1
}

step_fit() {  # step_fit <config> <tag> [extra --set args...]
    local config="$1" tag="$2"; shift 2
    if [ -n "$(latest_run "$tag")" ] && [ -f "$(latest_run "$tag")/best.pth" ]; then
        echo "== $tag: already done ($(latest_run "$tag")), skipping =="
        return
    fi
    echo "== $tag: training =="
    embedkd fit --config "$config" --set run.tag="$tag" "$@"
}

# --- 0. Teacher -------------------------------------------------------------
step_fit configs/d1_cub200_teacher.yaml d1_teacher_resnet50
TEACHER="$(latest_run d1_teacher_resnet50)/best.pth"
echo "Teacher checkpoint: $TEACHER"

# --- 1. Pre-distillation diagnostics (goes into the paper, demo D5) ---------
embedkd diagnose --config configs/d1_cub200_cosine.yaml \
    --set teacher.weights="$TEACHER" \
    --out "$(dirname "$TEACHER")/compatibility_report.json"

# --- 2. Four objectives, seed 42 ---------------------------------------------
for OBJ in cosine mse kl rkd; do
    step_fit "configs/d1_cub200_${OBJ}.yaml" "d1_${OBJ}_s42" \
        --set teacher.weights="$TEACHER"
done

# --- 3. Seed replicates for the main objective (tolerance estimation) --------
for SEED in 43 44; do
    step_fit configs/d1_cub200_cosine.yaml "d1_cosine_s${SEED}" \
        --set teacher.weights="$TEACHER" --set train.seed="$SEED"
done

# --- 4. Deployment benchmark for teacher vs cosine student -------------------
COSINE="$(latest_run d1_cosine_s42)/best.pth"
embedkd deploy --config configs/d1_cub200_cosine.yaml \
    --set teacher.weights="$TEACHER" \
    --checkpoint "$COSINE" --out-dir "$(dirname "$COSINE")/deploy"

# --- 5. Aggregate expected results -------------------------------------------
python scripts/make_expected_results.py d1_cub200 \
    --runs "d1_cosine_s42:cosine" "d1_mse_s42:mse" "d1_kl_s42:kl" "d1_rkd_s42:rkd" \
    --seed-runs d1_cosine_s42 d1_cosine_s43 d1_cosine_s44 \
    --config configs/d1_cub200_cosine.yaml \
    --checkpoint-tag d1_cosine_s42

echo "== D1 pipeline complete. See expected_results/d1_cub200.json =="
