#!/usr/bin/env bash
# CNN + 注意力：mask 只产生 Q/K，projector 仅接收更新后的视觉特征。
# 从原 LLaVA 底座重新 SFT，然后测试；不加载任何旧 adapter，不做 mask-only 预训练。
# 沿用旧阶段编号：3=SFT，4=测试。默认顺序执行 3-4。
# 运行：bash scripts/run_cnn_attention_pipeline.sh
# 预览：DRY_RUN=1 bash scripts/run_cnn_attention_pipeline.sh
# 双卡：TRAIN_GPUS=0,1 TEST_GPU=0 bash scripts/run_cnn_attention_pipeline.sh
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "${PROJECT_ROOT}"

# ====================== 配置区：路径均可包含空格 ======================
RUN_NAME="${RUN_NAME:-cnn_attention_only_v1}"
START_STAGE="${START_STAGE:-3}"
END_STAGE="${END_STAGE:-4}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov}"
FUSION_CONFIG="${FUSION_CONFIG:-configs/cnn_attention_only.json}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero3.json}"
SFT_OUTPUT="${SFT_OUTPUT:-outputs/${RUN_NAME}/sft}"
LOG_DIR="${LOG_DIR:-outputs/${RUN_NAME}/logs}"

# 一张卡用普通 Python，多张卡自动使用 ZeRO-3。编号逗号分隔，无空格。
TRAIN_GPUS="${TRAIN_GPUS:-6}"
TEST_GPU="${TEST_GPU:-6}"
# 沿用最近实验的 FP16 + FA2；可覆盖为 DTYPE=bfloat16 或 ATTN_IMPLEMENTATION=sdpa。
DTYPE="${DTYPE:-float16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
SEED="${SEED:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_EVERY="${LOG_EVERY:-10}"
MAX_TEXT_LENGTH="${MAX_TEXT_LENGTH:-2048}"

TRAIN_FILES=(
    "/home/yz/xxy/data/datasets/Triad_train_data/hdd2/movable/InstructIAD/InstructIAD_musc_muscv2_classnorm_group_score.json"
    "/home/yz/xxy/data/datasets/Triad_train_data/hdd2/movable/CoTM/CoTM_musc_muscv2_classnorm_group_score.json"
)
VAL_FILES=()  # 可选真实验证集；不要填写最终测试集。
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/home/yz/xxy/data/datasets}"
VAL_DATA_ROOT="${VAL_DATA_ROOT:-${TRAIN_DATA_ROOT}}"
REQUIRE_DISJOINT_CATEGORIES="${REQUIRE_DISJOINT_CATEGORIES:-1}"
SFT_IMAGE_MODE="${SFT_IMAGE_MODE:-randomroi}"
SFT_EPOCHS="${SFT_EPOCHS:-3}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-4}"
SFT_LR="${SFT_LR:-1e-4}"
SFT_PROJECTOR_LR="${SFT_PROJECTOR_LR:-1e-5}"

TEST_DATA_ROOT="${TEST_DATA_ROOT:-/home/yz/xxy/data/datasets}"
TEST_DATA="${TEST_DATA:-${TEST_DATA_ROOT}/Traid_eval_data/mvtec/question_musc_muscv2_classnorm_group_score.jsonl}"
TEST_OUTPUT="${TEST_OUTPUT:-results/mvtec_${RUN_NAME}.jsonl}"
ANOMALY_PROMPT="${ANOMALY_PROMPT:-category-mcq}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
VISUALIZE="${VISUALIZE:-1}"
VISUALIZATION_LIMIT="${VISUALIZATION_LIMIT:-0}"
VISUALIZATION_DIR="${VISUALIZATION_DIR:-results/mvtec_${RUN_NAME}_vis}"
# ====================== 配置区结束 ======================

fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }
(( $# == 0 )) || fail 'Edit the configuration above or set environment variables; CLI arguments are not accepted.'
[[ "${START_STAGE}" =~ ^[34]$ && "${END_STAGE}" =~ ^[34]$ ]] || fail 'Only stages 3 (real-image SFT) and 4 (test) are supported; mask-only pretraining is disabled.'
(( START_STAGE <= END_STAGE )) || fail 'START_STAGE must not exceed END_STAGE.'
for flag in DRY_RUN VISUALIZE REQUIRE_DISJOINT_CATEGORIES; do
    [[ "${!flag}" =~ ^[01]$ ]] || fail "${flag} must be 0 or 1."
done
selected() { (( START_STAGE <= $1 && $1 <= END_STAGE )); }
require_file() { [[ -f "$1" ]] || fail "File not found: $1"; }

train_world_size=1
if selected 3; then
    (( ${#TRAIN_FILES[@]} > 0 )) || fail 'TRAIN_FILES must contain training manifests.'
    [[ "${TRAIN_GPUS}" =~ ^[^,[:space:]]+(,[^,[:space:]]+)*$ ]] || fail 'TRAIN_GPUS must be comma-separated GPU IDs without spaces.'
    IFS=, read -r -a gpu_ids <<< "${TRAIN_GPUS}"
    train_world_size=${#gpu_ids[@]}
    seen_ids=()
    for gpu_id in "${gpu_ids[@]}"; do
        for seen_id in "${seen_ids[@]}"; do
            [[ "${gpu_id}" != "${seen_id}" ]] || fail 'TRAIN_GPUS must select different GPUs.'
        done
        seen_ids+=("${gpu_id}")
    done
fi
if selected 4; then
    [[ -n "${TEST_GPU}" && "${TEST_GPU}" != *,* && "${TEST_GPU}" != *[[:space:]]* ]] || fail 'TEST_GPU must select one GPU.'
fi
if [[ "${DRY_RUN}" == "0" ]]; then
    command -v "${PYTHON_BIN}" >/dev/null || fail "Python not found: ${PYTHON_BIN}"
    [[ -d "${BASE_MODEL}" ]] || fail "Base model directory not found: ${BASE_MODEL}"
    if selected 3; then
        require_file "${FUSION_CONFIG}"
        for manifest in "${TRAIN_FILES[@]}" "${VAL_FILES[@]}"; do require_file "${manifest}"; done
        if (( train_world_size > 1 )); then require_file "${DEEPSPEED_CONFIG}"; fi
    fi
    if selected 4; then require_file "${TEST_DATA}"; fi
fi

export OMP_NUM_THREADS PYTHONUNBUFFERED=1
current_stage=setup
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
    "$@" 2>&1 | tee -a -- "${LOG_DIR}/${current_stage}.log"
}
printf 'Project: %s\nRun: %s\nStages: %s-%s (fresh CNN attention-only SFT)\n' "${PROJECT_ROOT}" "${RUN_NAME}" "${START_STAGE}" "${END_STAGE}"

launcher=(env "CUDA_VISIBLE_DEVICES=${TRAIN_GPUS}" "${PYTHON_BIN}" -m)
distributed_args=(--distributed none --device cuda:0)
if (( train_world_size > 1 )); then
    launcher+=(torch.distributed.run --standalone "--nproc_per_node=${train_world_size}" --module)
    distributed_args=(--distributed deepspeed --deepspeed-config "${DEEPSPEED_CONFIG}")
fi
validation_args=()
if (( ${#VAL_FILES[@]} > 0 )); then
    validation_args+=(--val-data "${VAL_FILES[@]}" --val-data-root "${VAL_DATA_ROOT}")
    if [[ "${REQUIRE_DISJOINT_CATEGORIES}" == "1" ]]; then validation_args+=(--require-disjoint-categories); fi
fi
run_stage 3 sft \
    "${launcher[@]}" mask_fusion.train "${distributed_args[@]}" \
    --base-model "${BASE_MODEL}" --fusion-config "${FUSION_CONFIG}" --training-stage sft \
    --data "${TRAIN_FILES[@]}" --data-root "${TRAIN_DATA_ROOT}" "${validation_args[@]}" \
    --output "${SFT_OUTPUT}" --image-mode "${SFT_IMAGE_MODE}" \
    --dtype "${DTYPE}" --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --epochs "${SFT_EPOCHS}" --batch-size "${SFT_BATCH_SIZE}" --grad-accum "${SFT_GRAD_ACCUM}" \
    --lr "${SFT_LR}" --projector-lr "${SFT_PROJECTOR_LR}" \
    --num-workers "${NUM_WORKERS}" --log-every "${LOG_EVERY}" \
    --max-text-length "${MAX_TEXT_LENGTH}" --seed "${SEED}"

visualization_args=()
if [[ "${VISUALIZE}" == "1" ]]; then
    visualization_args+=(--visualize --visualization-dir "${VISUALIZATION_DIR}" --visualization-limit "${VISUALIZATION_LIMIT}")
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
    printf '\nDRY_RUN finished: no training/test started and no outputs created.\n'
else
    printf '\nStages %s-%s completed. Logs: %s\n' "${START_STAGE}" "${END_STAGE}" "${LOG_DIR}"
    if selected 4; then printf 'Predictions: %s\n' "${TEST_OUTPUT}"; fi
fi
