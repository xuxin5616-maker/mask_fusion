#!/usr/bin/env bash
# Jet/CNN 双通路 adapter：仅替换直接进入 projector 的专家特征，顺序测试四组。
# 原 mask、mask 引导 ROI、Q/K/V 和视觉残差保持原样；不修改任何训练权重。
# 运行：bash scripts/test_projector_expert_ablation.sh
# 预览：DRY_RUN=1 bash scripts/test_projector_expert_ablation.sh
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "${PROJECT_ROOT}"

# ===================== 路径和参数 =====================
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov}"
DATA_ROOT="${DATA_ROOT:-/home/yz/xxy/data/datasets}"
TEST_DATA="${TEST_DATA:-${DATA_ROOT}/Traid_eval_data/mvtec/question_musc_muscv2_classnorm_group_score.jsonl}"
JET_ADAPTER="${JET_ADAPTER:-outputs/siglip_jet_v1/sft}"
CNN_ADAPTER="${CNN_ADAPTER:-outputs/mask_residual_mlp_sft_bf16_fa2_gpu6_fixroot/sft}"
TEST_GPU="${TEST_GPU:-6}"
# 沿用这两个已发结果的精度；DTYPE 可统一覆盖两者。
JET_DTYPE="${JET_DTYPE:-${DTYPE:-float16}}"
CNN_DTYPE="${CNN_DTYPE:-${DTYPE:-bfloat16}}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
ANOMALY_PROMPT="${ANOMALY_PROMPT:-category-mcq}"
SEED="${SEED:-0}"  # 所有实验使用相同的数据/ROI seed
NOISE_SEED="${NOISE_SEED:-0}"
NOISE_MEAN="${NOISE_MEAN:-0}"
NOISE_STD="${NOISE_STD:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-results/projector_expert_ablation_v1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MAX_TEXT_LENGTH="${MAX_TEXT_LENGTH:-2048}"
# 设为 1 时，每个模型额外跑真实专家特征对照，共六组。
RUN_ORIGINAL="${RUN_ORIGINAL:-0}"
# 可选保存输入图；它仍显示原始 mask，替换发生在高维特征层。
VISUALIZE="${VISUALIZE:-0}"
VISUALIZATION_LIMIT="${VISUALIZATION_LIMIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
# ===================== 配置结束 =======================

fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }
(( $# == 0 )) || fail 'Set paths/parameters above or through environment variables; no CLI arguments.'
for flag in DRY_RUN RUN_ORIGINAL VISUALIZE; do
    [[ "${!flag}" =~ ^[01]$ ]] || fail "${flag} must be 0 or 1."
done
[[ -n "${TEST_GPU}" && "${TEST_GPU}" != *,* && "${TEST_GPU}" != *[[:space:]]* ]] || fail 'TEST_GPU must select one GPU.'
[[ "${NOISE_SEED}" =~ ^[0-9]+$ ]] || fail 'NOISE_SEED must be a nonnegative integer.'
[[ "${VISUALIZATION_LIMIT}" =~ ^[0-9]+$ ]] || fail 'VISUALIZATION_LIMIT must be nonnegative.'
if [[ "${DRY_RUN}" == "0" ]]; then
    command -v "${PYTHON_BIN}" >/dev/null || fail "Python not found: ${PYTHON_BIN}"
    [[ -d "${BASE_MODEL}" ]] || fail "Base model not found: ${BASE_MODEL}"
    [[ -f "${TEST_DATA}" ]] || fail "Test manifest not found: ${TEST_DATA}"
    for adapter in "${JET_ADAPTER}" "${CNN_ADAPTER}"; do
        [[ -f "${adapter}/adapter_config.json" && -f "${adapter}/adapter.safetensors" ]] || fail "Adapter not found: ${adapter}"
    done
    [[ ! -e "${OUTPUT_DIR}" ]] || fail "Output directory already exists: ${OUTPUT_DIR}; choose a new OUTPUT_DIR."
    mkdir -p -- "${OUTPUT_DIR}/logs"
fi

modes=(zero random)
if [[ "${RUN_ORIGINAL}" == "1" ]]; then modes=(original zero random); fi
metrics_files=()
for model in jet cnn; do
    if [[ "${model}" == "jet" ]]; then adapter="${JET_ADAPTER}"; dtype="${JET_DTYPE}"
    else adapter="${CNN_ADAPTER}"; dtype="${CNN_DTYPE}"; fi
    for mode in "${modes[@]}"; do
        name="${model}_${mode}"
        if [[ "${mode}" == "random" ]]; then name+="_seed${NOISE_SEED}"; fi
        output="${OUTPUT_DIR}/${name}.jsonl"
        command=(env "CUDA_VISIBLE_DEVICES=${TEST_GPU}" "${PYTHON_BIN}" -m mask_fusion.infer_expert_ablation
            --expert-input "${mode}" --ablation-label "${model}"
            --noise-seed "${NOISE_SEED}" --noise-mean "${NOISE_MEAN}" --noise-std "${NOISE_STD}"
            --adapter "${adapter}" --base-model "${BASE_MODEL}"
            --data "${TEST_DATA}" --data-root "${DATA_ROOT}"
            --eval-task anomaly --anomaly-prompt "${ANOMALY_PROMPT}"
            --output "${output}" --device cuda:0 --dtype "${dtype}"
            --attn-implementation "${ATTN_IMPLEMENTATION}" --seed "${SEED}"
            --max-new-tokens "${MAX_NEW_TOKENS}" --max-text-length "${MAX_TEXT_LENGTH}")
        if [[ "${VISUALIZE}" == "1" ]]; then
            command+=(--visualize --visualization-dir "${OUTPUT_DIR}/${name}_vis"
                --visualization-limit "${VISUALIZATION_LIMIT}")
        fi
        printf '\n[%s] ' "${name}"; printf '%q ' "${command[@]}"; printf '\n'
        if [[ "${DRY_RUN}" == "0" ]]; then
            "${command[@]}" 2>&1 | tee "${OUTPUT_DIR}/logs/${name}.log"
        fi
        metrics_files+=("${output%.jsonl}.metrics.json")
    done
done
summary_command=("${PYTHON_BIN}" -m mask_fusion.infer_expert_ablation
    --summarize "${metrics_files[@]}" --summary-output "${OUTPUT_DIR}/comparison")
printf '\n'; printf '%q ' "${summary_command[@]}"; printf '\n'
if [[ "${DRY_RUN}" == "0" ]]; then "${summary_command[@]}"; fi
