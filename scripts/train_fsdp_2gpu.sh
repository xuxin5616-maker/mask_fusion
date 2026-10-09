#!/usr/bin/env bash
# 双 A100 启动脚本：在下面数组填写训练/验证 JSON 或 JSONL 路径（每个路径加引号）。
# 示例：TRAIN_FILES=("/data/train_a.jsonl" "/data/my dataset/train_b.json")
# VAL_FILES 可留空；有验证集时默认检查训练/验证类别不重叠。
# 运行：bash scripts/train_fsdp_2gpu.sh
# 或：bash scripts/train_fsdp_2gpu.sh --data train_a.jsonl train_b.jsonl --val-data val.jsonl
# 预览：DRY_RUN=1 bash scripts/train_fsdp_2gpu.sh --data "/data/train file.jsonl"
# 关闭类别隔离检查：REQUIRE_DISJOINT_CATEGORIES=0；相对路径以项目目录为基准。
# 其他训练参数可直接追加，例如 --resume outputs/fsdp_2gpu/checkpoint-epoch-1。
set -euo pipefail

TRAIN_FILES=()
VAL_FILES=()

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "${PROJECT_ROOT}"

BASE_MODEL="${BASE_MODEL:-/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/fsdp_2gpu}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# 只检查是否提供了数据选项；实际参数解析交给 mask_fusion.train。
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
    printf '%s\n' '请填写脚本顶部 TRAIN_FILES=("训练文件路径" ...)，或传入 --data train1.jsonl train2.jsonl。' >&2
    exit 2
fi

command=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --standalone --nproc_per_node=2 --module mask_fusion.train
    --distributed fsdp
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
)
# 默认启用梯度检查点；需要关闭时在命令行传 --no-gradient-checkpointing。
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
