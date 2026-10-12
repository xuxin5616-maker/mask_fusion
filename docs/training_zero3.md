# 双卡 ZeRO-3 训练与 ETA

现有 `mask_fusion.train` 新增 `--distributed deepspeed`，原有单卡与 FSDP 入口继续可用。
参考 `Triad/Triad/finetune_ov_0shot.sh` 的 torchrun + ZeRO-3 方式以及
`Triad/Triad/LLaVA-NeXT/scripts/zero3.json`，只训练融合模块和整个 projector。
vision tower、Qwen LLM 保持冻结。

## 安装和启动

在 Linux CUDA 环境中，保留当前可运行的 PyTorch/Transformers 环境，安装额外后端：

```bash
python -m pip install -r requirements-deepspeed.txt
```

新脚本 `scripts/train_zero3_2gpu.sh` 默认使用 GPU 0、1，BF16、batch size 1、
每卡累积 4 批次，完整累积组的有效 batch size 为 8。
默认 `configs/no_hidden_norm.json` 关闭 mask 编码器 H 的 LayerNorm；E 的 LayerNorm 保留。
默认输出为 `outputs/zero3_2gpu_no_hidden_norm`。

填写脚本中的 `TRAIN_FILES`、可选 `VAL_FILES`，并指定正确的 DATA_ROOT；也可以直接传：

```bash
bash scripts/train_zero3_2gpu.sh \
  --data /absolute/path/train_a.json /absolute/path/train_b.jsonl \
  --data-root /absolute/path/to/image_and_mask_root
```

JSON 内 `image`、`mask` 是绝对路径时不受 DATA_ROOT 影响；相对路径拼到 DATA_ROOT 后应能找到文件。
Bash 数组中多个路径用空格分隔，不加逗号。训练数据支持多个 JSON/JSONL。
有验证文件时默认检查类别隔离；关闭方式同 FSDP：`REQUIRE_DISJOINT_CATEGORIES=0`。
先用 `DRY_RUN=1` 查看命令，不会加载模型或训练。
可覆盖 `--fusion-config`、`--output` 等参数。

## ETA 显示

- 单卡、FSDP 和 ZeRO-3 都在第一个成功更新、每 `--log-every` 个优化器更新和每轮末尾输出训练进度。
- 日志中的 `elapsed` 是本次运行已用训练时间，`eta` 是剩余所有 epoch 的训练批次估计时间，格式为 `小时:分钟:秒`。
- 训练 ETA 不包含模型初始化、验证、保存 checkpoint 的时间。验证单独输出 `validation_progress` 及 ETA。
- 断点续训按恢复后的实际处理速度估算；不会把以前完成的批次当作本次测得的速度。
- 推理每条输出类似 `[17/100] elapsed=00:01:25 ETA=00:06:55 ...`，包含当前循环中的可视化和结果写入时间。
  推理指标文件新增 `inference_progress`。推理命令无需增加参数。
- 前几批、不同 crop 数、问题长度和生成长度都可能使估计波动。
- 更频繁显示训练 ETA：追加 `--log-every 1`，或新脚本使用 `LOG_EVERY=1`。

## 分片和重复加载日志

torchrun 为每张 GPU 启动一个进程。每个进程读入同一基础 checkpoint，日志可能出现两次。
这不表示每张卡在训练时常驻完整模型。新入口先在 CPU 加载，再调用 `zero.Init(module=...)`
逐参数搬运、分片，之后创建 ZeRO-3 引擎。启动日志 `zero3_initialized` 显示：

- `global_parameter_elements`：全模型参数元素数。
- `rank0_partition_elements`：rank 0 的分片元素数（包含对齐 padding）。
- `cuda_allocated_gib`：初始化完成时 rank 0 的 PyTorch 显存分配量。

CPU 加载仍需要每进程一份模型内存。训练时 ZeRO-3 按需临时聚合当前模块参数，
激活、通信缓存、持久化的小参数也占显存，因此显存占用不是简单精确除以二。
这套配置与 Triad 一样不启用 CPU/NVMe offload；此入口会拒绝 offload 配置。

DeepSpeed 的 BF16/FP16 计算参数与 FP32 优化器主权重由引擎管理。
原 FSDP 的融合模块参数保持 FP32 并使用 autocast；两种后端不保证逐位相同的结果。
JSON 中的 batch、累积步数、精度、裁剪阈值由 CLI 解析 `auto`；显式冲突会报错。
实际解析后的配置保存在输出目录的 `deepspeed_config.json`。

## 保存、恢复、推理

每个 checkpoint 包括现有推理格式的 `adapter.safetensors`、`adapter_config.json`、tokenizer，
以及 `deepspeed/state/` 下的优化器分片和 `training_state.pt` 恢复元数据。
冻结的 LLM/vision tower 权重从原始 base checkpoint 读取，不重复保存。
融合模块中未启用的分支仍完整导出，保持推理加载器的严格键校验。

```bash
bash scripts/train_zero3_2gpu.sh \
  --data /absolute/path/train_a.json /absolute/path/train_b.jsonl \
  --data-root /absolute/path/to/image_and_mask_root \
  --resume outputs/zero3_2gpu_no_hidden_norm/checkpoint-epoch-1
```

恢复时保持数据、卡数、精度、epochs、batch、学习率、融合配置与原运行一致。
必须保留完整 checkpoint 目录，不能只有 adapter 文件。
从 FSDP 切换 ZeRO-3 应使用新输出目录重新训练；FSDP 优化器状态不兼容 ZeRO-3 的 `--resume`。

原推理命令只需把 `--adapter` 指向新输出目录，其余用法不变。
推理不依赖 DeepSpeed。原始 LLaVA 权重仍通过 `--base-model` 提供。

实现参考：[DeepSpeed ZeRO-3 参数协调](https://deepspeed.readthedocs.io/en/latest/zero3.html)、
[训练引擎](https://deepspeed.readthedocs.io/en/latest/training.html)、
[checkpoint API](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html)。

本地 CPU 测试只能验证控制流、数值累积、保存接口和脚本参数；完整多卡 NCCL/ZeRO-3 训练需在 Linux GPU 机器验证。
