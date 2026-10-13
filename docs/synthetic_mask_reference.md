# 合成 mask 的灰度参考

更新日期：2026-10-13。参考输入为 `outputs/nor/` 和 `outputs/abn/` 中各 5 张用户标注的 PNG。重点是灰度深浅；形状采用相同的椭圆、块状和细条族作粗略近似。MVTec AD 包含划痕、凹陷、污染等缺陷，但这里没有复制真实缺陷 mask，也不声称重现其完整分布。[MVTec AD 原论文](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_ad.pdf)

## 样例观察

全部参考是 RGB PNG，尺寸约 700×700；其中 9 张通道完全相同，另一张少量彩色像素位于边缘。统计剔除四边各 3% 以减少截图边框影响；这不是精确内容分割。以下范围是每张图指标的最小至最大值，只代表显示灰度，不是原始 MuSc V2 概率的标定。

| 指标 | nor，5 张 | abn，5 张 |
|---|---:|---:|
| 平均灰度 | 12.9–28.3 | 44.8–78.5 |
| 灰度第 99 百分位 | 30–50 | 117–197 |
| 最大灰度 | 35–54 | 127–216 |
| 灰度大于 64 的像素比例 | 0% | 24.5%–64.5% |
| 缩放为 384 后最高完整 14×14 patch 均分 | 33.2–50.7 | 120.9–201.2 |

本地可复核的产物位于 `outputs/mask_reference_analysis_20261013/`：`diagnostics.json` 包含逐图、原尺寸与裁边后的统计，`comparison_contact_sheet.png` 为固定灰度显示的对照图，`analyze_references.py` 为分析脚本。输出目录不随 Git 提交。

关键反例是 `abn/69349a71-8a92-4d78-808b-012a15c2b1df.png`：用户标为异常，裁边后最大灰度仅 127。旧规则要求像素达到 128 且最高 patch 均分达到 140，会直接漏掉这种中灰异常。旧 normal 配置中的 190–250 高亮点也不符合这组正常参考。

## 当前生成配置

配置文件：`configs/synthetic_masks.json`。

| 项目 | normal | abnormal |
|---|---:|---:|
| 背景基值采样范围 | 3–20 | 10–40 |
| 背景噪声标准差采样范围 | 2–6 | 3–10 |
| 区域目标峰值采样范围 | 25–55 | 95–210 |
| 区域数量 | 1–3 | 1–3 |
| 区域提议框面积比例 | 0.5%–35% | 0.5%–35% |
| 区域边缘模糊半径 | 5–15 | 5–15 |
| 额外孤立高亮噪点 | 关闭 | 关闭 |

这是生成参数，不是输出统计的硬限制：叠加背景、噪声、模糊和形状支持后，最终最大值/面积会改变。所有输出统计和 QA 答案都从保存后的 PNG 重算。输出为单通道 0–255，保留数值尺度，不做逐图 min-max。两类共用形状及面积选择，区别优先由灰度范围提供；这些是近似扩充范围，没有对仅 10 张样例做精确分布拟合。

异常标签采用 `blurred_peak`。先裁取模型实际编码的左上 378×378，再用 Pillow 对 uint8 分数图做半径 3 的高斯平滑，平滑峰值 >=80 为合成 abnormal，否则 normal。该步骤只计算监督标签，不改变保存的训练输入；没有独立的最小亮区面积条件。局部平滑降低单像素尖峰的影响，但不保证保留所有极小异常。

80 处于这组正常/异常显示灰度间隔内，是合成数据的起始规则，**不是实际推理阈值，不是用 10 张图验证过的通用分类器**。coverage/shape 题使用原分数 >=64 的区域；它们与 anomaly 题的规则不同。

旧规则与旧灰度范围保存在 `configs/synthetic_masks_generic.json`，供复现实验。

## 生成新数据并训练

```bash
python -m mask_fusion.synthetic_masks \
  --config configs/synthetic_masks.json \
  --output data/synthetic_masks_reference \
  --train-count 2000 --val-count 200 \
  --size 384 --seed 0 --preview-count 16

SYNTHETIC_DATA_DIR=data/synthetic_masks_reference \
bash scripts/pretrain_mask_zero3_2gpu.sh --dtype float32
```

先检查预览再做长时间训练；2000/200 是命令示例，不是数据量结论。修改配置不会更新已有数据文件，新分布请使用新目录。训练仍沿用 mask-only 第一阶段，第二阶段仍通过 `--init-adapter` 接入真实图像和原来的 QA 数据。

本次已生成 100 张训练及 20 张验证 mask（共 600 条 QA）用于预览，位于 `outputs/synthetic_mask_reference_preview_20261013/`。全项目 218 项 CPU 测试通过，包括弱异常、小亮块、孤立噪点、边界不可见像素和旧规则兼容；没有运行 7B/GPU 预训练或验证实际检测收益。
