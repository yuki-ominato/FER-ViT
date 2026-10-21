#!/usr/bin/env bash
# =============================================================================
# AFS StyleExtractor の学習 → 成分抽出 → 可視化評価 → LatentViTv2 学習 を一括で行う。
#
# 抽出器の種類 (EXTRACTOR_TYPE):
#   paper : train/train_style_extractor.py（AFS 論文の再現。ArcFace + LPIPS + 32x32 特徴）
#   fer   : train/train_fer_extractor.py  （FER 特化。画像ベース FER 損失など）
#
# 作成元の記録:
#   抽出器チェックポイント    → ckpt['provenance'] と <run_dir>/provenance.json
#   抽出済み latent           → <出力ディレクトリ>/provenance.json
#   LatentViTv2 の学習 config → data.train_provenance / data.val_provenance
#   抽出済み latent は ${EXTRACT_ROOT}/${EXTRACTOR_TYPE}_${RUN_ID}/ に置くので、
#   ディレクトリ名からも抽出器の run が分かる。
#
# 使い方（リポジトリルートで実行。変数は環境変数で上書きする）:
#   # RAF-DB e4e・論文再現版で全段階を実行
#   bash scripts/run_afs_pipeline.sh
#
#   # FER 特化版（画像ベース FER モデルを指定）
#   EXTRACTOR_TYPE=fer FER_IMAGE_CKPT=experiments/.../best_model.pt bash scripts/run_afs_pipeline.sh
#
#   # 学習済み抽出器から抽出以降だけ実行
#   STAGES="extract eval fit" EXTRACTOR_CKPT=outputs/extractor/raf-db_e4e/paper/20261021_120000/checkpoints/best_model.pt \
#       bash scripts/run_afs_pipeline.sh
#
#   # pSp latent で実行
#   ENCODER=psp LATENT_ROOT=latents/raf-db PSP_PATH=pretrained_models/psp_ffhq_encode.pt \
#       bash scripts/run_afs_pipeline.sh
#
#   # 実行せずにコマンドだけ確認
#   DRY_RUN=1 bash scripts/run_afs_pipeline.sh
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

# ---- データ・エンコーダ -------------------------------------------------------
: "${DATASET:=raf-db}"                      # raf-db | fer2013（実験名にのみ使う）
: "${ENCODER:=e4e}"                         # e4e | psp（latent の生成元。実験名にのみ使う）
: "${LATENT_ROOT:=latents/raf-db_e4e}"      # ${LATENT_ROOT}/${TRAIN_SPLIT}, ${VAL_SPLIT} を使う
: "${TRAIN_SPLIT:=train}"
: "${VAL_SPLIT:=test}"                      # RAF-DB は val split が無いため現状は test（best 選択にも使われる点に注意）
: "${PSP_PATH:=pretrained_models/e4e_ffhq_encode.pt}"   # StyleGAN2 decoder の取得元。latent と同じエンコーダにする
: "${ARCFACE_PATH:=pretrained_models/model_ir_se50.pth}"

# ---- 抽出器 -------------------------------------------------------------------
: "${EXTRACTOR_TYPE:=paper}"                # paper | fer
: "${EXTRACTOR_EPOCHS:=10}"
: "${EXTRACTOR_BATCH_SIZE:=4}"
: "${EXTRACTOR_LR:=1e-4}"
: "${FER_IMAGE_CKPT:=}"                     # fer のときの画像ベース FER モデル（空なら L_expr_img 等は無効）
: "${EXTRACTOR_CKPT:=}"                     # 指定すると train 段階を飛ばしてこのチェックポイントを使う
: "${RUN_ID:=$(date +%Y%m%d_%H%M%S)}"
: "${EXTRACTOR_OUT:=outputs/extractor/${DATASET}_${ENCODER}/${EXTRACTOR_TYPE}}"
: "${EXTRACT_ROOT:=latents/afs/${DATASET}_${ENCODER}}"

# ---- LatentViTv2 学習 ---------------------------------------------------------
: "${COMPONENTS:=style identity}"           # style: h(w) / identity: w - h(w)
: "${RUN_BASELINE:=1}"                      # 1 なら抽出なしの元 latent でも同条件で学習する
: "${FRACTIONS:=1.0 0.1}"
: "${SEEDS:=42 43 44}"
: "${FIT_EPOCHS:=60}"
: "${FIT_BATCH_SIZE:=64}"
: "${FIT_LR:=1e-4}"
: "${EXP_ROOT:=afs_rerun/${DATASET}_${ENCODER}}"   # experiments/ 以下の実験名の接頭辞

: "${STAGES:=train extract eval fit}"
: "${DRY_RUN:=0}"
export CUBLAS_WORKSPACE_CONFIG=:16:8

run() {
    echo "+ $*"
    if [[ "${DRY_RUN}" != "1" ]]; then
        "$@"
    fi
}

has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }

# ---- 抽出器チェックポイントの決定 ----------------------------------------------
if [[ -n "${EXTRACTOR_CKPT}" ]]; then
    # 既存チェックポイントを使う場合は run ディレクトリ名を RUN_ID として引き継ぐ
    RUN_ID="$(basename "$(dirname "$(dirname "${EXTRACTOR_CKPT}")")")"
    STAGES="${STAGES//train/}"
    # 抽出器の種類はチェックポイントの記録（旧形式は run の config.json）から判定し、指定値より優先する
    if [[ -f "${EXTRACTOR_CKPT}" ]]; then
        detected="$(python -c 'import sys; from afs.provenance import describe_extractor_checkpoint as d; print(d(sys.argv[1])["extractor_type"])' "${EXTRACTOR_CKPT}" | tail -n 1)"
        if [[ "${detected}" != "${EXTRACTOR_TYPE}" ]]; then
            echo "NOTE: チェックポイントの記録に従い EXTRACTOR_TYPE を ${EXTRACTOR_TYPE} → ${detected} に変更します"
        fi
        EXTRACTOR_TYPE="${detected}"
    fi
else
    EXTRACTOR_CKPT="${EXTRACTOR_OUT}/${RUN_ID}/checkpoints/best_model.pt"
fi
EXTRACT_DIR="${EXTRACT_ROOT}/${EXTRACTOR_TYPE}_${RUN_ID}"

echo "============================================================"
echo " DATASET=${DATASET} ENCODER=${ENCODER} EXTRACTOR_TYPE=${EXTRACTOR_TYPE} RUN_ID=${RUN_ID}"
echo " EXTRACTOR_CKPT=${EXTRACTOR_CKPT}"
echo " EXTRACT_DIR=${EXTRACT_DIR}"
echo " STAGES=${STAGES}"
echo "============================================================"

# ---- 1. 抽出器の学習 -------------------------------------------------------------
if has_stage train; then
    common_args=(
        --latent_dir     "${LATENT_ROOT}/${TRAIN_SPLIT}"
        --val_latent_dir "${LATENT_ROOT}/${VAL_SPLIT}"
        --psp_path       "${PSP_PATH}"
        --arcface_path   "${ARCFACE_PATH}"
        --out_dir        "${EXTRACTOR_OUT}"
        --run_id         "${RUN_ID}"
        --epochs         "${EXTRACTOR_EPOCHS}"
        --batch_size     "${EXTRACTOR_BATCH_SIZE}"
        --lr             "${EXTRACTOR_LR}"
    )
    if [[ "${EXTRACTOR_TYPE}" == "paper" ]]; then
        run python train/train_style_extractor.py "${common_args[@]}" --provider a
    elif [[ "${EXTRACTOR_TYPE}" == "fer" ]]; then
        fer_args=()
        if [[ -n "${FER_IMAGE_CKPT}" ]]; then
            fer_args=(--fer_image_ckpt "${FER_IMAGE_CKPT}")
        else
            echo "WARNING: FER_IMAGE_CKPT が空のため L_expr_img / L_neutral_img は無効になります" >&2
        fi
        run python train/train_fer_extractor.py "${common_args[@]}" "${fer_args[@]}"
    else
        echo "EXTRACTOR_TYPE は paper か fer を指定してください: ${EXTRACTOR_TYPE}" >&2
        exit 1
    fi
fi

# ---- 2. 成分抽出（train / val split それぞれ style と identity を保存） -------------
if has_stage extract; then
    for split in "${TRAIN_SPLIT}" "${VAL_SPLIT}"; do
        run python data/extract_style_latents.py \
            --latent_dir           "${LATENT_ROOT}/${split}" \
            --out_dir              "${EXTRACT_DIR}/${split}" \
            --style_extractor_path "${EXTRACTOR_CKPT}" \
            --mode                 both
    done
fi

# ---- 3. 分解の可視化・定量評価（id_sim / cons / 7x7 cross-class grid） --------------
if has_stage eval; then
    run python eval/evaluate_afs_rafdb.py \
        --ckpt_path    "${EXTRACTOR_CKPT}" \
        --psp_path     "${PSP_PATH}" \
        --arcface_path "${ARCFACE_PATH}" \
        --latent_dir   "${LATENT_ROOT}/${VAL_SPLIT}"
fi

# ---- 4. LatentViTv2 の学習 --------------------------------------------------------
fit() {
    local train_dir="$1" val_dir="$2" name="$3" frac="$4" seed="$5"
    run python train/train_latent_vit_v2.py \
        --latent_train_dir "${train_dir}" \
        --latent_val_dir   "${val_dir}" \
        --data_fraction    "${frac}" \
        --seed             "${seed}" \
        --epochs           "${FIT_EPOCHS}" \
        --batch_size       "${FIT_BATCH_SIZE}" \
        --lr               "${FIT_LR}" \
        --experiment_name  "${name}"
}

if has_stage fit; then
    for frac in ${FRACTIONS}; do
        pct="$(awk -v f="${frac}" 'BEGIN { printf "%d", f * 100 + 0.5 }')"
        for seed in ${SEEDS}; do
            if [[ "${RUN_BASELINE}" == "1" ]]; then
                fit "${LATENT_ROOT}/${TRAIN_SPLIT}" "${LATENT_ROOT}/${VAL_SPLIT}" \
                    "${EXP_ROOT}/baseline/frac${pct}_seed${seed}" "${frac}" "${seed}"
            fi
            for comp in ${COMPONENTS}; do
                fit "${EXTRACT_DIR}/${TRAIN_SPLIT}/${comp}" "${EXTRACT_DIR}/${VAL_SPLIT}/${comp}" \
                    "${EXP_ROOT}/${EXTRACTOR_TYPE}_${RUN_ID}/${comp}/frac${pct}_seed${seed}" "${frac}" "${seed}"
            done
        done
    done
fi

echo "完了: RUN_ID=${RUN_ID}"
