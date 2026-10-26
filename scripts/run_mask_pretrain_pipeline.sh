#!/usr/bin/env bash
# 完整流程：1 生成数据 -> 2 mask 预训练 -> 3 真实数据 SFT -> 4 MVTec 测试。
# 先激活原来训练成功的 conda 环境；只需修改下方配置区。
# 运行：bash scripts/run_mask_pretrain_pipeline.sh
# 预览：DRY_RUN=1 bash scripts/run_mask_pretrain_pipeline.sh（不创建文件、不启动 Python）
# 已完成前两步：START_STAGE=3 bash scripts/run_mask_pretrain_pipeline.sh
# 单卡：TRAIN_GPUS=0 TEST_GPU=0 DTYPE=bfloat16 START_STAGE=2 bash scripts/run_mask_pretrain_pipeline.sh
# START_STAGE 只跳过已完成阶段，不会恢复阶段内中断的优化器状态。
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "${PROJECT_ROOT}"

# ====================== 配置区：路径均可包含空格 ======================
# 相对路径以项目目录为基准。新实验修改 RUN_NAME；保留旧结果，不自动覆盖。
RUN_NAME="${RUN_NAME:-siglip_jet_v1}"
START_STAGE="${START_STAGE:-1}"  # 1/2/3/4，从哪个阶段开始
END_STAGE="${END_STAGE:-4}"      # 例如 END_STAGE=1 只生成数据
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov}"
FUSION_CONFIG="${FUSION_CONFIG:-configs/siglip_jet.json}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero3.json}"
LOG_DIR="${LOG_DIR:-outputs/${RUN_NAME}/logs}"

# GPU 是 CUDA_VISIBLE_DEVICES 的编号。填一个用普通单卡训练；填多个用 ZeRO-3。
# 例如 TRAIN_GPUS=2 用 2 号卡，TRAIN_GPUS=2,3 用 2、3 号卡；测试只使用一张卡。
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
TEST_GPU="${TEST_GPU:-0}"
DTYPE="${DTYPE:-float32}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
SEED="${SEED:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_EVERY="${LOG_EVERY:-10}"
MAX_TEXT_LENGTH="${MAX_TEXT_LENGTH:-2048}"

# 第 1 阶段：合成 mask。数量是独立 mask 数，每张默认产生 5 条 QA。
SYNTHETIC_CONFIG="${SYNTHETIC_CONFIG:-configs/synthetic_masks.json}"
SYNTHETIC_DATA_DIR="${SYNTHETIC_DATA_DIR:-data/synthetic_masks_${RUN_NAME}}"
SYNTHETIC_TRAIN_COUNT="${SYNTHETIC_TRAIN_COUNT:-2000}"
SYNTHETIC_VAL_COUNT="${SYNTHETIC_VAL_COUNT:-200}"
MASK_SIZE="${MASK_SIZE:-384}"
PREVIEW_COUNT="${PREVIEW_COUNT:-16}"

# 第 2 阶段：RGB 特征置零；mask 在线转 Jet 后仍经冻结 SigLIP。
# 训练 A 的直通 pre-LN 残差 MLP 及 projector 新增输入列，不更新 SigLIP。
# square 保留完整 mask，不能改为 ROI 裁剪，否则几何题的答案可能失效。
PRETRAIN_OUTPUT="${PRETRAIN_OUTPUT:-outputs/${RUN_NAME}/pretrain}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-3}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-1}"
PRETRAIN_GRAD_ACCUM="${PRETRAIN_GRAD_ACCUM:-4}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-4}"
PRETRAIN_PROJECTOR_LR="${PRETRAIN_PROJECTOR_LR:-1e-4}"

# 第 3 阶段：原来的真实训练数据。多个路径用空格/换行分隔，不要加逗号。
TRAIN_FILES=(
    "/home/yz/xxy/data/datasets/Triad_train_data/hdd2/movable/InstructIAD/InstructIAD_musc_muscv2_classnorm_group_score.json"
    "/home/yz/xxy/data/datasets/Triad_train_data/hdd2/movable/CoTM/CoTM_musc_muscv2_classnorm_group_score.json"
)
VAL_FILES=()  # 可选真实验证集；不要填写最终测试集
# root 是 JSON 内相对 image/mask 路径的前缀；本数据包含 Triad_train_data/...。
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/home/yz/xxy/data/datasets}"
VAL_DATA_ROOT="${VAL_DATA_ROOT:-${TRAIN_DATA_ROOT}}"
REQUIRE_DISJOINT_CATEGORIES="${REQUIRE_DISJOINT_CATEGORIES:-1}"
SFT_OUTPUT="${SFT_OUTPUT:-outputs/${RUN_NAME}/sft}"
SFT_IMAGE_MODE="${SFT_IMAGE_MODE:-randomroi}"
SFT_EPOCHS="${SFT_EPOCHS:-3}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-4}"
SFT_LR="${SFT_LR:-1e-4}"
SFT_PROJECTOR_LR="${SFT_PROJECTOR_LR:-1e-5}"

# 第 4 阶段：测试自动读取 SFT_OUTPUT 的 image_mode，默认是 randomroi。
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/home/yz/xxy/data/datasets}"
TEST_DATA="${TEST_DATA:-${TEST_DATA_ROOT}/Traid_eval_data/mvtec/question_musc_muscv2_classnorm_group_score.jsonl}"
TEST_OUTPUT="${TEST_OUTPUT:-results/mvtec_${RUN_NAME}.jsonl}"
ANOMALY_PROMPT="${ANOMALY_PROMPT:-category-mcq}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
VISUALIZE="${VISUALIZE:-1}"
VISUALIZATION_LIMIT="${VISUALIZATION_LIMIT:-0}"  # 0 保存所有测试记录的可视化
VISUALIZATION_DIR="${VISUALIZATION_DIR:-results/mvtec_${RUN_NAME}_vis}"
# ====================== 配置区结束 ======================

fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }
if (( $# > 0 )); then
    fail 'Edit the configuration at the top, or set environment variables; positional/CLI arguments are not accepted.'
fi
[[ "${START_STAGE}" =~ ^[1-4]$ && "${END_STAGE}" =~ ^[1-4]$ ]] || fail 'START_STAGE and END_STAGE must be 1, 2, 3 or 4.'
(( START_STAGE <= END_STAGE )) || fail 'START_STAGE must not exceed END_STAGE.'
for flag in DRY_RUN VISUALIZE REQUIRE_DISJOINT_CATEGORIES; do
    [[ "${!flag}" =~ ^[01]$ ]] || fail "${flag} must be 0 or 1."
done

selected() { (( START_STAGE <= $1 && $1 <= END_STAGE )); }
require_file() { [[ -f "$1" ]] || fail "File not found: $1"; }

# 在正式启动前检查外部输入；DRY_RUN 不依赖数据、模型或 GPU 的存在。
train_world_size=1
if selected 2 || selected 3; then
    [[ "${TRAIN_GPUS}" =~ ^[^,[:space:]]+(,[^,[:space:]]+)*$ ]] || fail 'TRAIN_GPUS must contain one or more comma-separated GPU IDs without spaces (e.g. 0 or 0,1).'
    IFS=, read -r -a train_gpu_ids <<< "${TRAIN_GPUS}"
    train_world_size=${#train_gpu_ids[@]}
    seen_gpu_ids=()
    for gpu_id in "${train_gpu_ids[@]}"; do
        for seen_id in "${seen_gpu_ids[@]}"; do
            [[ "${gpu_id}" != "${seen_id}" ]] || fail 'TRAIN_GPUS must select different GPUs.'
        done
        seen_gpu_ids+=("${gpu_id}")
    done
fi
if selected 4; then
    [[ -n "${TEST_GPU}" && "${TEST_GPU}" != *,* && "${TEST_GPU}" != *[[:space:]]* ]] || fail 'TEST_GPU must select one GPU.'
fi
if selected 3; then
    (( ${#TRAIN_FILES[@]} > 0 )) || fail 'TRAIN_FILES must contain at least one training manifest.'
fi
if [[ "${DRY_RUN}" == "0" ]]; then
    command -v "${PYTHON_BIN}" >/dev/null || fail "Python not found: ${PYTHON_BIN}; activate your training environment first."
    if selected 1; then require_file "${SYNTHETIC_CONFIG}"; fi
    if (( END_STAGE >= 2 )); then
        [[ -d "${BASE_MODEL}" ]] || fail "Base model directory not found: ${BASE_MODEL}"
    fi
    if selected 2 || selected 3; then
        require_file "${FUSION_CONFIG}"
        if (( train_world_size > 1 )); then require_file "${DEEPSPEED_CONFIG}"; fi
    fi
    if selected 3; then
        for manifest in "${TRAIN_FILES[@]}" "${VAL_FILES[@]}"; do require_file "${manifest}"; done
    fi
    if selected 4; then require_file "${TEST_DATA}"; fi
fi

export OMP_NUM_THREADS PYTHONUNBUFFERED=1
current_stage='setup'
trap 'status=$?; printf "Stage %s failed (exit %s); later stages were not started.\n" "${current_stage}" "${status}" >&2; exit "${status}"' ERR

run_stage() {
    local number="$1" name="$2"
    shift 2
    if ! selected "${number}"; then return 0; fi
    current_stage="${number}_${name}"
    printf '\n[%s/4] %s\n' "${number}" "${name}"
    printf '%q ' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
    mkdir -p -- "${LOG_DIR}"
    # pipefail 保证 Python/torchrun 失败时整个脚本退出；追加日志保留先前失败信息。
    "$@" 2>&1 | tee -a -- "${LOG_DIR}/${current_stage}.log"
}

printf 'Project: %s\nRun: %s\nStages: %s-%s\n' "${PROJECT_ROOT}" "${RUN_NAME}" "${START_STAGE}" "${END_STAGE}"

run_stage 1 generate \
    "${PYTHON_BIN}" -m mask_fusion.synthetic_masks \
    --config "${SYNTHETIC_CONFIG}" --output "${SYNTHETIC_DATA_DIR}" \
    --train-count "${SYNTHETIC_TRAIN_COUNT}" --val-count "${SYNTHETIC_VAL_COUNT}" \
    --size "${MASK_SIZE}" --seed "${SEED}" --preview-count "${PREVIEW_COUNT}"

distributed_command=(
    env "CUDA_VISIBLE_DEVICES=${TRAIN_GPUS}" "${PYTHON_BIN}" -m
)
training_args=(
    --base-model "${BASE_MODEL}" --fusion-config "${FUSION_CONFIG}"
    --dtype "${DTYPE}" --attn-implementation "${ATTN_IMPLEMENTATION}"
    --num-workers "${NUM_WORKERS}" --log-every "${LOG_EVERY}"
    --max-text-length "${MAX_TEXT_LENGTH}" --seed "${SEED}"
)
if (( train_world_size == 1 )); then
    # 普通单卡入口不初始化分布式进程组，也不依赖 DeepSpeed。
    training_args+=(--distributed none --device cuda:0)
else
    distributed_command+=(torch.distributed.run --standalone "--nproc_per_node=${train_world_size}" --module)
    training_args+=(--distributed deepspeed --deepspeed-config "${DEEPSPEED_CONFIG}")
fi
run_stage 2 pretrain \
    "${distributed_command[@]}" mask_fusion.pretrain "${training_args[@]}" \
    --data "${SYNTHETIC_DATA_DIR}/train.jsonl" \
    --val-data "${SYNTHETIC_DATA_DIR}/val.jsonl" \
    --data-root "${SYNTHETIC_DATA_DIR}" --val-data-root "${SYNTHETIC_DATA_DIR}" \
    --output "${PRETRAIN_OUTPUT}" --image-mode square \
    --epochs "${PRETRAIN_EPOCHS}" --batch-size "${PRETRAIN_BATCH_SIZE}" \
    --grad-accum "${PRETRAIN_GRAD_ACCUM}" --lr "${PRETRAIN_LR}" \
    --projector-lr "${PRETRAIN_PROJECTOR_LR}"

validation_args=()
if (( ${#VAL_FILES[@]} > 0 )); then
    validation_args+=(--val-data "${VAL_FILES[@]}" --val-data-root "${VAL_DATA_ROOT}")
    if [[ "${REQUIRE_DISJOINT_CATEGORIES}" == "1" ]]; then
        validation_args+=(--require-disjoint-categories)
    fi
fi
run_stage 3 sft \
    "${distributed_command[@]}" mask_fusion.train "${training_args[@]}" \
    --training-stage sft --init-adapter "${PRETRAIN_OUTPUT}" \
    --data "${TRAIN_FILES[@]}" --data-root "${TRAIN_DATA_ROOT}" "${validation_args[@]}" \
    --output "${SFT_OUTPUT}" --image-mode "${SFT_IMAGE_MODE}" \
    --epochs "${SFT_EPOCHS}" --batch-size "${SFT_BATCH_SIZE}" \
    --grad-accum "${SFT_GRAD_ACCUM}" --lr "${SFT_LR}" \
    --projector-lr "${SFT_PROJECTOR_LR}"

visualization_args=()
if [[ "${VISUALIZE}" == "1" ]]; then
    visualization_args+=(--visualize --visualization-dir "${VISUALIZATION_DIR}"
                         --visualization-limit "${VISUALIZATION_LIMIT}")
fi
run_stage 4 test \
    env "CUDA_VISIBLE_DEVICES=${TEST_GPU}" "${PYTHON_BIN}" -m mask_fusion.infer \
    --adapter "${SFT_OUTPUT}" --base-model "${BASE_MODEL}" \
    --data "${TEST_DATA}" --data-root "${TEST_DATA_ROOT}" \
    --eval-task anomaly --anomaly-prompt "${ANOMALY_PROMPT}" --output "${TEST_OUTPUT}" \
    --device cuda:0 --dtype "${DTYPE}" --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --seed "${SEED}" --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-text-length "${MAX_TEXT_LENGTH}" "${visualization_args[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '\nDRY_RUN finished: no data generated, no training/test started, no logs created.\n'
else
    printf '\nStages %s-%s completed. Logs: %s\n' "${START_STAGE}" "${END_STAGE}" "${LOG_DIR}"
    if selected 4; then printf 'Predictions: %s\n' "${TEST_OUTPUT}"; fi
fi
