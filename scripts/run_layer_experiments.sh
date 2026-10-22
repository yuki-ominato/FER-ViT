#!/usr/bin/env bash
# =============================================================================
# 層ごとの表情情報の分析と LayerMix 拡張の比較を一括で行う。test は best 選択に使わない。
#
#   split      : train latent からクラス層化の holdout（検証用）を切り出す
#   probe      : 層ごとの線形分類（holdout で評価。CPU のみ・数分）
#   fit        : LatentViTv2 を baseline / LayerMix で学習（holdout で best 選択、最後に test を 1 度評価）
#   importance : baseline の best モデルで層アブレーション（holdout で評価）
#
# 層番号は 0 始まり。LayerMix は MIX_KEEP の層を残し、それ以外をバッチ内の別サンプルと入れ替える。
#
# 使い方（リポジトリルートで実行。変数は環境変数で上書きする）:
#   bash scripts/run_layer_experiments.sh                          # RAF-DB pSp で全段階
#   STAGES="split probe" bash scripts/run_layer_experiments.sh     # GPU 不要な段階だけ
#   ENCODER=e4e LATENT_ROOT=latents/raf-db_e4e bash scripts/run_layer_experiments.sh
#   DRY_RUN=1 bash scripts/run_layer_experiments.sh                # コマンドの確認のみ
#
# 結果:
#   test の値      : experiments/${EXP_ROOT}/.../experiment_summary.json の final_metrics.test_*
#   holdout の値   : 同 best_f1_macro
#   集計           : python scripts/collect_experiments.py
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

: "${DATASET:=raf-db}"
: "${ENCODER:=psp}"
: "${LATENT_ROOT:=latents/raf-db}"          # ${LATENT_ROOT}/${TRAIN_SPLIT}, ${TEST_SPLIT}
: "${TRAIN_SPLIT:=train}"
: "${TEST_SPLIT:=test}"
: "${HOLDOUT_FRACTION:=0.1}"
: "${SPLIT_SEED:=0}"
# 分割はファイル名で持つので、同じデータセットなら pSp / e4e / AFS で共通に使える
: "${SPLIT_JSON:=splits/${DATASET}_holdout$(awk -v f="${HOLDOUT_FRACTION}" 'BEGIN { printf "%d", f * 100 + 0.5 }')_seed${SPLIT_SEED}.json}"

: "${FRACTIONS:=1.0 0.1}"
: "${SEEDS:=42 43 44}"
: "${MIX_PROBS:=0.5}"                       # 空白区切りで複数指定すると確率ごとに学習する
: "${MIX_KEEP:=4-11}"
: "${FIT_EPOCHS:=60}"
: "${FIT_BATCH_SIZE:=64}"
: "${FIT_LR:=1e-4}"
: "${EXP_ROOT:=layer_mix/${DATASET}_${ENCODER}}"
: "${PROBE_OUT:=eval_results/layer_probe/${DATASET}_${ENCODER}}"
: "${IMPORTANCE_OUT:=eval_results/layer_importance/${DATASET}_${ENCODER}}"

: "${STAGES:=split probe fit importance}"
: "${DRY_RUN:=0}"
export CUBLAS_WORKSPACE_CONFIG=:16:8

run() {
    echo "+ $*"
    if [[ "${DRY_RUN}" != "1" ]]; then
        "$@"
    fi
}

has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }
pct() { awk -v f="$1" 'BEGIN { printf "%d", f * 100 + 0.5 }'; }

TRAIN_DIR="${LATENT_ROOT}/${TRAIN_SPLIT}"
TEST_DIR="${LATENT_ROOT}/${TEST_SPLIT}"

echo "============================================================"
echo " DATASET=${DATASET} ENCODER=${ENCODER} LATENT_ROOT=${LATENT_ROOT}"
echo " SPLIT_JSON=${SPLIT_JSON}"
echo " STAGES=${STAGES}"
echo "============================================================"

# ---- 1. holdout 分割 ----------------------------------------------------------
if has_stage split; then
    if [[ -f "${SPLIT_JSON}" ]]; then
        echo "既存の分割を使います: ${SPLIT_JSON}"
    else
        run python data/holdout_split.py --latent_dir "${TRAIN_DIR}" --out "${SPLIT_JSON}" \
            --val_fraction "${HOLDOUT_FRACTION}" --seed "${SPLIT_SEED}"
    fi
fi

# ---- 2. 層ごとの線形分類 -------------------------------------------------------
if has_stage probe; then
    for frac in ${FRACTIONS}; do
        run python eval/layer_probe.py --latent_dir "${TRAIN_DIR}" --holdout_split "${SPLIT_JSON}" \
            --output_dir "${PROBE_OUT}" --data_fraction "${frac}"
    done
fi

# ---- 3. LatentViTv2 の学習（baseline / LayerMix） --------------------------------
fit() {
    local name="$1" frac="$2" seed="$3"; shift 3
    run python train/train_latent_vit_v2.py \
        --latent_train_dir "${TRAIN_DIR}" \
        --latent_val_dir   "${TEST_DIR}" \
        --holdout_split    "${SPLIT_JSON}" \
        --data_fraction    "${frac}" \
        --seed             "${seed}" \
        --epochs           "${FIT_EPOCHS}" \
        --batch_size       "${FIT_BATCH_SIZE}" \
        --lr               "${FIT_LR}" \
        --experiment_name  "${name}" \
        "$@"
}

if has_stage fit; then
    for frac in ${FRACTIONS}; do
        for seed in ${SEEDS}; do
            fit "${EXP_ROOT}/baseline/frac$(pct "${frac}")_seed${seed}" "${frac}" "${seed}"
            for p in ${MIX_PROBS}; do
                fit "${EXP_ROOT}/mix_keep${MIX_KEEP}_p${p}/frac$(pct "${frac}")_seed${seed}" "${frac}" "${seed}" \
                    --layer_mix_prob "${p}" --layer_mix_keep "${MIX_KEEP}"
            done
        done
    done
fi

# ---- 4. 層アブレーション（baseline の best モデル、holdout で評価） ----------------
if has_stage importance; then
    for frac in ${FRACTIONS}; do
        for seed in ${SEEDS}; do
            name="${EXP_ROOT}/baseline/frac$(pct "${frac}")_seed${seed}"
            # 同名の実験が複数あれば最新の run を使う
            ckpt="$(ls -1d experiments/${name}/*/checkpoints/best_model.pt 2>/dev/null | sort | tail -n 1 || true)"
            if [[ -z "${ckpt}" ]]; then
                if [[ "${DRY_RUN}" == "1" ]]; then
                    ckpt="experiments/${name}/<run>/checkpoints/best_model.pt"
                else
                    echo "WARNING: checkpoint が見つかりません: experiments/${name}" >&2
                    continue
                fi
            fi
            run python eval/analyze_layer_importance.py --checkpoint_path "${ckpt}" \
                --latent_test_dir "${TRAIN_DIR}" --holdout_split "${SPLIT_JSON}" \
                --output_dir "${IMPORTANCE_OUT}/frac$(pct "${frac}")_seed${seed}"
        done
    done
fi

echo "完了"
