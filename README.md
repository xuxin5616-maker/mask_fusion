# LLaVA Mask Fusion

基于工作区 `llava_pruning` 的 LLaVA-OneVision / Qwen2 / SigLIP 路径，实现连续专家异常图的双通路融合。新增代码在 `mask_fusion/`；原模型运行时保留在 `vendor/llava/`。

当前入口支持单张原图、多视图（AnyRes/ROI）和多条 QA，训练支持单 GPU 或双 GPU FSDP SFT；也能在 CPU 上运行小模型测试。MuSc V2 作为外部冻结专家，输入它预先生成的连续分数图。本项目不包含 MuSc V2 的生成器、基础模型权重或真实训练数据。

## 模型结构

```mermaid
flowchart LR
    I[原图及对应裁剪] --> VT[冻结 SigLIP]
    VT --> X[图像特征 X]
    M[连续 mask 及对应裁剪] --> CNN[可训练 mask CNN]
    CNN --> H[H: 128维]
    M --> S[patch 均值与最大值]
    H --> A[A: 130维]
    S --> A
    A --> U[MLP: U 256维]
    P[固定二维位置编码128维] --> R[线性映射: R 256维]
    U --> E[通道拼接 E: 512维]
    R --> E
    E --> QK[独立 Q/K 投影]
    X --> V[V 投影]
    QK --> ATT[多头注意力]
    V --> ATT
    ATT --> RES[视觉残差 X + alpha deltaX]
    X --> RES
    RES --> CAT[通道拼接]
    A --> CAT
    CAT --> PROJ[扩展并微调原 projector]
    PROJ --> LLM[冻结 Qwen LLM]
```

单个 crop 的默认数据路径如下，省略 batch 维。视觉维度和 patch 大小从加载的 tower 读取；表内数字对应仓库的 SigLIP so400m-patch14-384。

| 阶段 | 计算 | 输出 |
|---|---|---|
| 视觉特征 | `VisionTower(image)` | `729 × 1152` |
| 固定分数缩放 | `mask / 255`，不逐图 min-max、不二值化 | `1 × 384 × 384` |
| mask 编码器 | `Conv(1,128,k14,s14) → GELU → Conv(128,128,k3,s1,p1) → GELU` | `128 × 27 × 27` |
| H | 展平行优先 patch 网格，可选逐 token LayerNorm | `729 × 128` |
| A | `[H; patch_mean; patch_max]` | `729 × 130` |
| U | `Linear(130,256) → GELU → Linear(256,256)` | `729 × 256` |
| R | 行/列整数坐标分别正余弦编码，再 `Linear(128,256)` | `729 × 256` |
| E | **`[U;R]`，无额外压缩线性层** | `729 × 512` |
| 注意力 | 默认对 E 做 LayerNorm，再独立 Q/K 投影；V 来自 X | 8 heads，每头 32 维 |
| 视觉残差 | `X_tilde = X + alpha * W_O(attention(Q,K,V))` | `729 × 1152` |
| projector 输入 | `[X_tilde;A]` | `729 × 1282` |
| projector 输出 | 原有 LLM hidden size，继续原 token 打包流程 | token 数量沿用 baseline |

公式中的位置编码、均值与最大值无可训练参数。CNN、U/R 映射、Q/K/V/O、融合系数和整个 projector 可训练。冻结 vision tower、Qwen 和外部专家；损失仍通过冻结 Qwen 反传到新模块。

默认只用一层注意力，不额外加入视觉 FFN。增加 `num_layers` 时，每层 Q/K 都来自相同的纯 mask/位置先验，V 来自上一层更新后的视觉特征。注意力在每个 crop 内独立计算。

配置见 `configs/default.json`。128/256 维宽度、1 层、8 个头、`alpha_init=0.01`、H/E LayerNorm 都是可修改的起始设置，不是已验证的最优超参数。`normalize_e=false` 可关闭注意力先验的归一化；直接进入 projector 的均值/最大值不经过 H/E 的 LayerNorm。

### 初始化与消融

projector 第一层复制旧视觉列、偏置和所有后续层，新增 130 列初始化为零。注意力使用正常初始化、alpha 从小非零值开始，因此完整模型初始输出不保证与 baseline 严格相同。

四种模式保持相同的扩展 projector 形状：

| config `mode` | projector 输入 | 可训练分支 |
|---|---|---|
| `both` | `[X_tilde;A]` | 全部新增模块 + projector |
| `direct` | `[X;A]` | mask CNN + projector |
| `attention` | `[X_tilde;0]` | mask CNN、先验和注意力 + projector |
| `none` | `[X;0]` | projector，作为相同 SFT 下的图像基线 |

每种模式应从相同原始 checkpoint 独立训练。对应配置为 `configs/default.json`、`direct.json`、`attention.json`、`none.json`。训练数据、裁剪、精度和后端应保持一致。`randomroi` 即使在 none 模式也可能用 mask 选框；若比较“不使用任何专家信息”，请统一使用 `anyres_max_9` 或 `square`。

## 环境

从本项目目录运行命令。真实训练建议使用已有 CUDA 环境，先安装适合硬件的 PyTorch，再安装其余依赖：

```bash
python -m pip install -r requirements.txt
```

接口沿用 baseline 的 `transformers==4.46.1`。默认 decoder 后端为 `sdpa`；如果需要匹配原 `llava_pruning` 的 FlashAttention 2 运行方式，请在安装匹配的 `flash-attn` 后显式设置 `--attn-implementation flash_attention_2`。不要把不同后端/精度的结果当作严格相同的基线运行。

本地验证环境 `.venv` 使用 Python 3.12、PyTorch 2.5.1 CPU、Transformers 4.46.1。它用于测试，不提供 CUDA 训练能力。

## 数据格式

训练、验证、推理共用 JSONL（一行一个对象），也支持 JSON 数组。`image`、`mask` 路径相对 JSON 文件目录解析，或通过 `--data-root` 指定统一根目录。

```json
{"id":"sample_001_q1","category":"bottle","image":"images/001.png","mask":"masks/001.npy","question":"Is there a visible defect?","options":{"A":"No visible defect","B":"A visible defect is present"},"answer":"B"}
```

- `options` 可用 A–H 连续字母的映射或 2–8 项字符串列表；训练答案可用字母或唯一匹配的选项文本，内部统一为字母。
- 普通开放 QA 省略 `options`，`answer` 为文本；推理可以省略 `answer`。
- `category` 用于验证未见产品类别和统计分类别结果。预检要求 `id` 和 `category` 完整。
- mask 支持灰度 PNG/TIFF、二维 NPY，或含 `anomaly_map` 的 NPZ。拒绝 RGB 热力图、NaN、负数和超出配置尺度的有效分数。
- 默认原值范围 0–255，融合模块统一除以 255；若源文件已经统一缩放到 0–1，请将 `score_scale` 显式设为 1，不能再除一次 255。
- `examples/train.jsonl` 仅说明格式，其图像和 mask 路径是待替换的示例，不是可直接训练的数据。

选择题 prompt 明确要求只回答选项字母。SFT 仅监督 assistant 答案和 ChatML 结束 token；system/user、图像和 padding 的标签均为 `-100`。超长文本或图像展开后超出上下文时直接报错，不静默截掉答案。

### 多个训练与验证文件

训练的 `--data`、`--val-data` 接受一个或多个文件，JSON 数组与 JSONL 可以混用；也可以重复提供同一个选项。文件按传入顺序合并为一个数据集，训练时统一打乱，不对每个文件单独训练或默认均衡采样。

```bash
python -m mask_fusion.train \
  --base-model /path/checkpoints/llava-onevision-qwen2-7b-ov \
  --data /path/product_a/train.jsonl /path/product_b/train.json \
  --val-data /path/product_c/val.jsonl /path/product_d/val.json \
  --require-disjoint-categories \
  --output outputs/multi_file
```

每个文件里的相对 `image`、`mask` 路径默认从该文件所在目录解析。例如 `/data/product_a/train.jsonl` 里的 `images/001.png` 对应 `/data/product_a/images/001.png`。需要统一数据根目录时再显式传 `--data-root`；验证集可使用 `--val-data-root`。同一文件内的 ID 必须唯一，不同文件可以有相同本地 ID；重复传入同一个文件会报错。

### 空间对应

默认 `anyres_max_9`，也支持 `square`、`randomroi`、`randompatch`。预处理复用 baseline 真正选出的 crop 和排列，不重新为 mask 抽样框。mask 使用 float32 双线性插值，RGB 使用原 SigLIP processor；相同几何变换不意味着共用 RGB 通道归一化。

patch 统计用真实 kernel/stride=14 的窗口，不能简单用 adaptive pooling 到 27×27 替代。padding 有独立有效性图：均值/最大值排除无效像素，完全无效 patch 不作为注意力 key；其专家特征和注意力增量置零。AnyRes 后续的 unpad、pooling、newline 打包沿用 baseline。

## 先检查跨产品类别划分

```bash
python -m mask_fusion.validate_data --train /path/data/train.jsonl --val /path/data/val.jsonl --test /path/data/test.jsonl --check-pixels
```

默认拒绝跨集合图像路径共享、产品类别共享、同图类别矛盾。统计分别报告 QA 数、独立图像数和类别数。图片副本/近重复图的内容去重仍需在数据整理阶段完成；路径检查不能发现不同文件名的同图。

预检的每个划分也支持多个文件，检查范围是整个合并后的划分，报告同时包含各源文件的统计：

```bash
python -m mask_fusion.validate_data \
  --train /path/product_a/train.jsonl /path/product_b/train.json \
  --val /path/product_c/val.jsonl /path/product_d/val.json \
  --test /path/product_e/test.jsonl \
  --check-pixels
```

## 训练

### 用脚本启动双卡训练

服务器上建议使用 `scripts/train_fsdp_2gpu.sh`。先编辑脚本顶部的两个数组，每行一个文件路径；没有验证集时保留 `VAL_FILES=()`：

```bash
TRAIN_FILES=(
  "/home/yz/xxy/data/qa/train_a.jsonl"
  "/home/yz/xxy/data/qa/train_b.json"
)
VAL_FILES=(
  "/home/yz/xxy/data/qa/val.jsonl"
)
```

以上数据路径是占位示例，需要换成自己的文件。保存后在项目根目录执行：

```bash
bash scripts/train_fsdp_2gpu.sh
```

也可以不编辑数组，直接传参；命令行数据列表会替换脚本中对应的数组：

```bash
bash scripts/train_fsdp_2gpu.sh \
  --data /path/train_a.jsonl /path/train_b.json \
  --val-data /path/val_a.jsonl /path/val_b.json \
  --output outputs/experiment_01
```

脚本默认使用 GPU `0,1`、BF16、每卡 batch size 1、梯度累积 4、3 个 epoch、融合模块学习率 `1e-4`、projector 学习率 `1e-5`，默认输出到 `outputs/fsdp_2gpu`。原始权重路径已设置为 `/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov`。脚本切换到项目根目录后运行，因此命令行的相对文件路径以项目根目录为准；数据记录中的相对图像/mask 路径仍按上文的数据规则解析。

可用 `CUDA_VISIBLE_DEVICES=2,3` 切换两张卡、`PYTHON_BIN=/path/env/bin/python` 指定 Python、`BASE_MODEL=/path/checkpoint` 修改底座路径，也可以追加训练参数，例如 `--fusion-config configs/direct.json`、`--save-every-steps 100` 或 `--resume outputs/fsdp_2gpu/checkpoint-epoch-1`。有验证集时脚本默认检查训练与验证的产品类别不重叠，符合未见类别评估目标；如需关闭此检查，可设置 `REQUIRE_DISJOINT_CATEGORIES=0`。

正式运行前可以预览最终命令，预览不会加载模型或启动训练，也不检查数据内容：

```bash
DRY_RUN=1 bash scripts/train_fsdp_2gpu.sh \
  --data /path/train_a.jsonl /path/train_b.json
```

### 单卡训练入口

```bash
python -m mask_fusion.train --base-model /path/checkpoints/llava-onevision-qwen2-7b-ov --data /path/data/train.jsonl --val-data /path/data/val.jsonl --require-disjoint-categories --output /path/runs/mask_fusion_both --fusion-config configs/default.json --image-mode anyres_max_9 --batch-size 1 --grad-accum 8 --epochs 3 --lr 1e-4 --projector-lr 1e-5
```

`--base-model` 必须是本地原始 LLaVA-Qwen checkpoint。若其配置中的 SigLIP 路径在当前机器不可用，可加 `--vision-tower /path/checkpoints/siglip-so400m-patch14-384`。不要换成不同结构/权重的视觉塔来加载同一个训练 adapter。

默认启用非重入 gradient checkpointing。`--dtype auto` 在支持时使用 BF16，否则使用 FP32；也可以明确选择 FP16。冻结骨干按选定精度加载，新增模块与 projector 参数保留 FP32，使用 autocast 计算。训练仍需要保存/重算冻结 LLM 的激活，不等同于只运行一个小 CNN 的显存成本。

保存内容：

- `adapter.safetensors`：完整训练后的 projector 和新增融合模块。
- `adapter_config.json`：融合结构、原始 checkpoint 路径、视觉维度和图像模式。
- `tokenizer/`：使用的 tokenizer。
- `training_state.pt`：优化器、学习率调度、随机状态与训练游标，用于显式恢复。
- `training_args.json`：运行参数；每个 epoch 另保存 checkpoint，有验证集时保存最低验证 loss 的 `best/`。

训练结束的输出目录本身可用于推理。恢复中断训练时复用原命令参数并添加：

```bash
--resume /path/runs/mask_fusion_both/checkpoint-epoch-1
```

恢复会检查数据清单和关键超参数一致性。多个文件会逐一记录路径和内容哈希，并保留文件顺序；增删文件、改变顺序或修改任一文件内容后，不能沿用原断点继续训练。它是继续原训练，不是用旧优化器开始另一份数据的新实验。只对自己保存、可信的训练状态使用 resume。

### 两张 A100 40GB：FSDP 分片训练

单卡入口保持不变。两张卡使用 `torchrun` 和 `--distributed fsdp`，将模型参数分片到两张卡；仅设置 `CUDA_VISIBLE_DEVICES=0,1` 不会把原单进程训练变成双卡训练。当前实现使用 PyTorch FSDP `FULL_SHARD`，不需要安装 DeepSpeed。

以下命令在服务器项目根目录执行。模型路径对应原 LLaVA-OneVision/Qwen2 checkpoint；数据路径需替换。

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  --module mask_fusion.train \
  --distributed fsdp \
  --base-model /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov \
  --data /path/to/qa_data/train.jsonl \
  --val-data /path/to/qa_data/val.jsonl \
  --data-root /path/to/qa_data \
  --require-disjoint-categories \
  --fusion-config configs/default.json \
  --image-mode anyres_max_9 \
  --output runs/mask_fusion_fsdp \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --batch-size 1 \
  --grad-accum 4 \
  --epochs 3 \
  --lr 1e-4 \
  --projector-lr 1e-5 \
  --log-every 10 \
  --save-every-steps 100
```

`batch-size` 是每张卡的 QA 数，因此正常累积组的总 batch 为 `2 × 1 × 4 = 8`，与原单卡 `1 × 8` 相同；若设置 `--grad-accum 8` 则为 16。最后不足一个累积组时按实际组大小缩放。训练数据为保证各 rank 相同更新步数会补齐，补齐可能重复极少数 QA；验证结果需排除补齐样本的重复贡献。

每个 Qwen decoder block、视觉塔、融合模块和 projector 是独立的分片单元，避免将 BF16 冻结参数和 FP32 可训练参数合并成同一 flat parameter。原权重先留在 CPU，再由 FSDP 逐单元迁移分片，避免初始化时先在每卡放完整模型。模型仍只训练原先指定的新模块和整个 projector，冻结的 Qwen 仍传播输入梯度。梯度裁剪使用 FSDP 的全局范数，梯度累积时每个 microbatch 都同步，避免 `no_sync` 保留全量梯度的额外显存。

FSDP 不会把两张 40GB 变成一块直接可用的 80GB：每卡仍需承受当前单元展开的参数、通信缓冲和本卡样本的激活。AnyRes 输入可能产生较长序列，需先在少量有代表性的样本上验证峰值显存。各进程也会在 CPU 读取原 checkpoint，服务器必须有足够主机内存。

所有 rank 参与必要的权重和优化器状态收集，仅 rank 0 写 checkpoint。导出的 `adapter.safetensors` 格式与单卡一致，可以继续用原 `mask_fusion.infer` 在单卡推理。续训使用相同卡数和原超参数，并在双卡命令后添加 `--resume runs/mask_fusion_fsdp/checkpoint-step-100`；不能把单卡优化器状态直接当成双卡训练续训状态。

在服务器运行以下小模型自检，无需下载真实模型权重或准备数据，检查两张 GPU 的梯度更新、冻结参数、adapter 单卡重载和优化器恢复。输出目录需为空或不存在：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  tests/fsdp_smoke.py --output runs/fsdp_smoke
```

该脚本通过只说明小模型的 CUDA/NCCL 分布式链路可运行，之后仍需用真实 7B 和少量不同裁剪数的 QA 检查显存及完整数据加载。

## 推理与选择题评估

```bash
python -m mask_fusion.infer --adapter /path/runs/mask_fusion_both/best --base-model /path/checkpoints/llava-onevision-qwen2-7b-ov --data /path/data/test.jsonl --output /path/results/predictions.jsonl
```

原 checkpoint 未移动时可省略 `--base-model`。图像模式从 adapter 自动恢复。推理为逐 QA 的贪心生成，默认最多 32 个新 token；开放问答可用 `--max-new-tokens` 增大。

`--base-model` 覆盖只应用于同一原始 checkpoint 的路径迁移。adapter 会检查结构和维度，但不对大模型权重逐文件做内容校验；同结构的另一套权重不能视作等价底座。

输出包括原始回答、解析出的选项和正确与否；旁边的 `.metrics.json` 报告总准确率、各类别准确率、类别宏平均和无法解析的选择题数量。普通文本 QA 保留输出但不自动判定语义正确性。

选择题解析器接受独立选项字母或精确的选项文本，不从任意长句中猜一个字母。无法解析的有标注选择题计为错误并单独报告。

## 验证与当前边界

```bash
python -m unittest discover -s tests -v
```

测试涵盖连续 mask 的真实 crop/patch 对齐、有效像素统计、位置编码与 E 拼接、注意力 padding、可训练参数与梯度、答案标签屏蔽、断点恢复、adapter 往返以及真实 tiny Qwen 的前向和生成。tiny Qwen 随机初始化，视觉塔用无下载的小型替身；这些测试验证代码行为，不验证异常检测准确率。

2026-10-08 验证结果：68 项 CPU 测试全部通过；训练、推理、数据预检三个 CLI 的 `--help` 均通过；新增代码、测试及 vendor Python 源码编译检查通过。FP16 overflow 的控制流程用模拟 scaler 注入溢出验证，尚不等同于真实 GPU FP16 验证。

2026-10-09：双卡训练改动后，83 项 CPU 回归/契约测试通过，训练 CLI `--help` 和 `compileall` 通过。新增验证覆盖混合精度冻结骨干梯度、alpha 存储形状不改变 BF16 输出与梯度、rank 0 导出无需额外 state-dict collective、分片数据补齐、验证统计、FSDP 梯度裁剪和集体保存接口。

同日，加入多文件与启动脚本后，全部 107 项测试通过。新增覆盖多源相对路径、JSON/JSONL 混用、文件内 ID 去重、跨文件划分检查、多文件训练的中途断点恢复，以及第二个文件内容变化后的恢复拒绝。启动脚本通过真实 Git Bash 的语法和 dry-run 参数测试，覆盖带空格路径、数组与命令行覆盖、环境变量和必填数据检查；这不等同于在 GPU 上执行训练。

同日，真实两个 CPU/Gloo 进程的 FSDP `FULL_SHARD` 小模型检查通过：同步梯度累积、末尾不足一个累积组、冻结参数不变、CNN/Q/alpha/projector 梯度、adapter 单设备重载以及优化器/调度器恢复后的下一步一致。CPU 路径需要关闭冗余的 state-dict CPU offload，避免本地 PyTorch 2.5.1 在原地 CPU 转换时复用存储；生产 CUDA 路径仍将完整 checkpoint 状态卸载到 CPU。这些 CPU 检查不等同于实际 CUDA/NCCL 双卡测试；后者使用上面的 `tests/fsdp_smoke.py`。

尚未用真实 7B checkpoint、GPU 和用户数据完成训练或评估。没有报告模型收益或显存/速度结论。当前 CLI 包含单 GPU 和可选 FSDP 分片训练入口，双 GPU 路径仍需在 Linux/NCCL/CUDA 服务器实际验证。支持单图 QA（含该图多个 crop），不包含普通 DDP、量化、视频或多张独立图像对话。

## 代码来源

`vendor/llava` 复制自工作区 `llava_pruning/vendor/llava`，源仓库提交 `75663414c497653d3bb5f4313a1c8af5da304c4c`。保留上游版权头及 `vendor/llava/LICENSE`。

复用：Qwen2/SigLIP、原 projector 结构、AnyRes/ROI 图像处理和视觉 token 打包。

新增：`mask_fusion/` 全部模块、配置、QA 数据入口、训练/推理、adapter 保存加载、数据预检和测试。

vendor 中只接线修改 `llava/model/llava_arch.py` 和 `llava/model/language_model/llava_qwen.py`：传递连续 mask/有效性图、在 vision tower 与 projector 之间调用新模块，并对超长序列显式报错。FastV/ViCo 代码保留，但该项目的 loader 禁用剪枝。
