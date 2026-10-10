#!/usr/bin/env bash
# MVTec AD 测试：沿用已训练 adapter 的图像模式，默认加入类别检测知识。
# 运行：bash scripts/test_mvtec.sh
# 预览命令（不加载模型）：DRY_RUN=1 bash scripts/test_mvtec.sh
# 原 A/B 对照：ANOMALY_PROMPT=category-mcq bash scripts/test_mvtec.sh
# 默认保存原图、实际 ROI/AnyRes 分块和连续 mask；VISUALIZE=0 可关闭。
# 只保存前 N 条：VISUALIZATION_LIMIT=20；默认 0 保存全部。
# 可用环境变量覆盖路径，也可追加 --adapter /path/to/adapter 等推理参数。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "${PROJECT_ROOT}"

DATA_ROOT="${DATA_ROOT:-/home/yz/xxy/data/datasets}"
TEST_DATA="${TEST_DATA:-${DATA_ROOT}/Traid_eval_data/mvtec/question_musc_muscv2_classnorm_group_score.jsonl}"
ADAPTER="${ADAPTER:-${PROJECT_ROOT}/outputs/fsdp_2gpu}"
BASE_MODEL="${BASE_MODEL:-/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov}"
ANOMALY_PROMPT="${ANOMALY_PROMPT:-category-knowledge-mcq}"
OUTPUT_FILE="${OUTPUT_FILE:-${PROJECT_ROOT}/results/mvtec_${ANOMALY_PROMPT//-/_}.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

command=(
    "${PYTHON_BIN}" -m mask_fusion.infer
    --adapter "${ADAPTER}"
    --base-model "${BASE_MODEL}"
    --data "${TEST_DATA}"
    --data-root "${DATA_ROOT}"
    --eval-task anomaly
    --anomaly-prompt "${ANOMALY_PROMPT}"
    --output "${OUTPUT_FILE}"
    --device cuda:0
    --dtype "${DTYPE:-bfloat16}"
    --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
    --seed "${SEED:-0}"
    --max-new-tokens "${MAX_NEW_TOKENS:-64}"
    --max-text-length "${MAX_TEXT_LENGTH:-2048}"
)
if [[ "${VISUALIZE:-1}" == "1" ]]; then
    command+=(--visualize)
fi
if [[ -n "${VISUALIZATION_DIR:-}" ]]; then
    command+=(--visualization-dir "${VISUALIZATION_DIR}")
fi
command+=(--visualization-limit "${VISUALIZATION_LIMIT:-0}")
command+=("$@")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'cd %q\n' "${PROJECT_ROOT}"
    printf 'CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q ' "${CUDA_VISIBLE_DEVICES}" "${OMP_NUM_THREADS}"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
