#!/usr/bin/env bash
# 双卡 mask-only 合成问答预训练；先生成 synthetic_masks 数据，再运行本脚本。
# 默认读取 data/synthetic_masks/{train,val}.jsonl，可用 SYNTHETIC_DATA_DIR 覆盖。
# 生成：python -m mask_fusion.synthetic_masks --output data/synthetic_masks --train-count 2000 --val-count 200 --size 384 --seed 0 --config configs/synthetic_masks.json --preview-count 16
# train-count/val-count 是独立 mask 数；默认合成配置未校准，训练前先查看生成预览并调整。
# 也可传 --data train1.jsonl train2.jsonl --val-data val.jsonl。
# DATA_ROOT 是 JSON 中相对 image/mask 路径的公共根目录；不设置时相对各 JSON。
# 预览（不要求数据、CUDA 或 DeepSpeed）：DRY_RUN=1 bash scripts/pretrain_mask_zero3_2gpu.sh
# 几何题必须使用 square；训练入口会拒绝裁剪模式以免改变标签含义。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "${PROJECT_ROOT}"

SYNTHETIC_DATA_DIR="${SYNTHETIC_DATA_DIR:-${PROJECT_ROOT}/data/synthetic_masks}"
TRAIN_FILES=("${SYNTHETIC_DATA_DIR}/train.jsonl")
VAL_FILES=("${SYNTHETIC_DATA_DIR}/val.jsonl")

BASE_MODEL="${BASE_MODEL:-/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/mask_pretrain_zero3}"
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

# Help is local and does not launch workers or require synthetic files.
if (( cli_help )) && [[ "${DRY_RUN:-0}" != "1" ]]; then
    exec "${PYTHON_BIN}" -m mask_fusion.pretrain "$@"
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    default_files=()
    if (( ! cli_train )); then
        default_files+=("${TRAIN_FILES[@]}")
    fi
    if (( ! cli_val )); then
        default_files+=("${VAL_FILES[@]}")
    fi
    for manifest in "${default_files[@]}"; do
        if [[ ! -f "${manifest}" ]]; then
            printf 'Synthetic manifest does not exist: %s\n' "${manifest}" >&2
            printf '%s\n' 'Generate the synthetic train/val manifests first, or pass --data and --val-data explicitly.' >&2
            printf 'Generate: %q -m mask_fusion.synthetic_masks --output %q --train-count 2000 --val-count 200 --size 384 --seed 0 --config %q --preview-count 16\n' "${PYTHON_BIN}" "${SYNTHETIC_DATA_DIR}" "${PROJECT_ROOT}/configs/synthetic_masks.json" >&2
            printf '%s\n' 'Inspect the generated previews and adjust the uncalibrated synthetic configuration before training.' >&2
            exit 2
        fi
    done
fi

command=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --standalone --nproc_per_node=2 --module mask_fusion.pretrain
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
    --projector-lr "${PROJECTOR_LR:-1e-4}"
    --image-mode square
    --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
    --num-workers "${NUM_WORKERS:-0}"
    --log-every "${LOG_EVERY:-10}"
)
if (( ! cli_train )); then
    command+=(--data "${TRAIN_FILES[@]}")
fi
if (( ! cli_val )); then
    command+=(--val-data "${VAL_FILES[@]}")
fi
if [[ -n "${DATA_ROOT:-}" ]]; then
    command+=(--data-root "${DATA_ROOT}")
fi
if [[ -n "${VAL_DATA_ROOT:-}" ]]; then
    command+=(--val-data-root "${VAL_DATA_ROOT}")
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
