#!/usr/bin/env bash
# 双卡 DeepSpeed ZeRO-3：填写数组，或在命令行传 --data 多个 JSON/JSONL。
# DATA_ROOT 是 JSON 内相对 image/mask 路径的共同前缀，不是 JSON 的目录。
# 示例：DATA_ROOT=/home/yz/xxy/data/datasets bash scripts/train_zero3_2gpu.sh --data train1.json train2.json
# 预览：DRY_RUN=1 bash scripts/train_zero3_2gpu.sh --data "train file.jsonl"
# 默认使用 default.json，启用 H 和 E 的 LayerNorm；可通过 --fusion-config 覆盖。
# 无 H LayerNorm 的旧实验可显式指定 configs/no_hidden_norm.json 和单独输出目录。
# FSDP checkpoint 不能恢复 ZeRO 优化器；首次切换请使用新输出目录。
set -euo pipefail

TRAIN_FILES=()
VAL_FILES=()

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "${PROJECT_ROOT}"

BASE_MODEL="${BASE_MODEL:-/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/zero3_2gpu_with_norm}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

cli_train=0
cli_val=0
cli_help=0
for argument in "$@"; do
    case "${argument}" in
        --data|--data=*) cli_train=1 ;;
        --val-data|--val-data=*|--validation|--validation=*) cli_val=1 ;;
        --help|-h) cli_help=1 ;;
    esac
done
if (( ! cli_train && ${#TRAIN_FILES[@]} == 0 && ! cli_help )); then
    printf '%s\n' '请填写 TRAIN_FILES=("训练文件路径" ...)，或传入 --data train1.jsonl train2.jsonl。' >&2
    exit 2
fi

command=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --standalone --nproc_per_node=2 --module mask_fusion.train
    --distributed deepspeed
    --deepspeed-config "${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/configs/zero3.json}"
    --fusion-config "${FUSION_CONFIG:-${PROJECT_ROOT}/configs/default.json}"
    --base-model "${BASE_MODEL}"
    --output "${OUTPUT_DIR}"
    --dtype bfloat16
    --batch-size "${BATCH_SIZE:-1}"
    --grad-accum "${GRAD_ACCUM:-4}"
    --epochs "${EPOCHS:-3}"
    --lr "${LR:-1e-4}"
    --projector-lr "${PROJECTOR_LR:-1e-5}"
    --image-mode "${IMAGE_MODE:-anyres_max_9}"
    --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
    --num-workers "${NUM_WORKERS:-0}"
    --log-every "${LOG_EVERY:-10}"
)
if (( ! cli_train && ${#TRAIN_FILES[@]} > 0 )); then
    command+=(--data "${TRAIN_FILES[@]}")
fi
if (( ! cli_val && ${#VAL_FILES[@]} > 0 )); then
    command+=(--val-data "${VAL_FILES[@]}")
fi
if [[ -n "${DATA_ROOT:-}" ]]; then
    command+=(--data-root "${DATA_ROOT}")
fi
if [[ -n "${VAL_DATA_ROOT:-}" ]]; then
    command+=(--val-data-root "${VAL_DATA_ROOT}")
fi
if (( cli_val || ${#VAL_FILES[@]} > 0 )) && [[ "${REQUIRE_DISJOINT_CATEGORIES:-1}" == "1" ]]; then
    command+=(--require-disjoint-categories)
fi
command+=("$@")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'cd %q\n' "${PROJECT_ROOT}"
    printf 'CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q ' "${CUDA_VISIBLE_DEVICES}" "${OMP_NUM_THREADS}"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
