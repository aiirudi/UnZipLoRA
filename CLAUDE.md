# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

UnZipLoRA：从单张图像中分离内容(content)和风格(style)的LoRA训练方法（ICCV 2025 Highlight）。基于SDXL，同时训练两个LoRA（content LoRA + style LoRA），通过三种分离策略（column separation、block separation、weight similarity）确保两个LoRA可通过直接相加合并。

## 环境配置

```bash
conda create -n unziplora python=3.11
conda activate unziplora
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

关键依赖：diffusers==0.25.0, transformers==4.42.3, xformers>=0.0.20

## 常用命令

训练：
```bash
bash train.sh
# 或直接用 accelerate launch train_unziplora.py --pretrained_model_name_or_path=... (参数见train.sh)
```

推理：
```bash
bash infer.sh
# 或 accelerate launch infer.py --output_dir=... --rank=64 --with_unziplora ...
```

交互探索：仓库提供 `playground.ipynb`、`load_mode.ipynb`、`load_mypipeline.ipynb` 用于加载已训练LoRA、检查权重/mask、调试pipeline。修改 `unziplora_unet/` 后建议先在notebook里复现一次加载和单步采样，再跑全量训练。

无独立测试套件。验证流程：用减少了 `--max_train_steps` 的 `train.sh` 跑一遍 → 用 `infer.sh` 推一张图 → 检查 `models/` 下产物和保存图像。

## 提示词约定

`train.sh` / `infer.sh` 都通过环境变量组合提示词：`CONTENT_RARE_TOKEN`（如`monadikos`）+ `CLASS_WORD`（subject类，如`cat`）+ `STYLE_RARE_TOKEN`（风格名）+ `SUPER_WORD`（用于style验证的不同subject类）。`PROMPT` / `CONTENT_FORWARD_PROMPT` / `STYLE_FORWARD_PROMPT` 三组提示词分别用于联合、content-only、style-only前向；`VALID_*` 系列覆盖各种验证组合。改提示词前理解这套约定，否则会破坏cross-attention对齐和验证图。

## 核心架构

### 训练流程 (`train_unziplora.py`)
- 基于 accelerate 的训练脚本，同时优化 content 和 style 两个 LoRA
- 使用三组独立学习率：content_learning_rate、style_learning_rate、weight_learning_rate
- 训练过程中交替进行：正常LoRA训练 → cone score累积 → column mask更新 → masked训练
- 输出产物：`{output_dir}_content/`（content LoRA）、`{output_dir}_style/`（style LoRA）、`{output_dir}_merger_content.pth`、`{output_dir}_merger_style.pth`（merge权重）。开 `--use_base_weight` 时还会保存 `{output_dir}_base_weight_content.pth` / `{output_dir}_base_weight_style.pth`。

### 推理流程 (`infer.py`)
`infer.py` 根据传入的 `--validation_prompt*` 参数选择不同 pipeline，所有路径都经过 `unziplora_unet/utils.py` 的注入器：
- `--validation_prompt` 非空 → `StableDiffusionXLUnZipLoRAPipeline`（`unziplora_unet/pipeline_stable_diffusion_xl.py`），双 hidden_states 前向，加载 `*_content` / `*_style` / `*_merger_*` / `*_base_weight_*`，用于合并 content + style。
- `--validation_prompt_content_recontext` 非空 → `StableDiffusionXLSingleLoRAPipeline` with `branch="content"`，只挂 content LoRA。
- `--validation_prompt_style` 非空 → 同上 but `branch="style"`。
- TFM 在 UnZip pipeline 中走 `prompt_embeds_content * rare_mask`（prompt embedding 层面置零），在 Single-LoRA pipeline 中走 `LoRALinearLayer.set_content_mask(...)`（LoRA up 之后乘 mask）；循环结束后 pipeline 会 `set_content_mask(None)` 清场，避免污染下一次推理。
- sigma_mask（`--use_time_control`）通过 `cross_attention_kwargs` 传递，由 `attention_processor.py` 的 `AttnProcessor2_0` 透传到 `LoRACompatibleLinear` → `UnZipLoRALinearLayerInfer` / `LoRALinearLayer`，在 LoRA 的 down 输出上逐列乘。

### UnZipLoRA层 (`unziplora_unet/unziplora_linear_layer.py`)
- `UnZipLoRALinearLayer`：训练用，包含 content/style 两组 down/up 矩阵 + merge向量 + mask
- `UnZipLoRALinearLayerInfer`：推理用的简化版本
- 每层维护：`lora_matrix_dic`(nn.ModuleDict存储down/up权重)、`merge_content/merge_style`(可学习软mask)、`mask_content/mask_style`(硬mask，bool)、`column_score_content/column_score_style`(cone得分)
- `forward_type`控制前向传播模式："both"/"content"/"style"

### 三种分离策略
1. **Column Separation**（`with_period_column_separation`）：基于cone score选择各LoRA激活的列子集，通过`mask_updated_elements`实现
2. **Block Separation**（`with_freeze_unet`）：UNet不同block分配给不同LoRA，配置在`SDXL_content_layer_mask`/`SDXL_style_layer_mask`
3. **Weight Similarity Loss**（`similarity_lambda`）：merge向量的余弦相似度惩罚，鼓励content/style使用不同列

### 关键术语（容易混）
代码里同时存在两套与 mask 相关的特性，名字相近，含义不同，必须分清：
- **TFM (Text Focus Mask)**：rare-token 文本对齐 mask。让 content 分支的 cross-attention 只作用于 rare token（如 `sks`）所在位置，从而把 content rare token 对齐到 class word。开关：训练 `--focus`（`train.sh` 里 `focus_value`），推理 `--focus_value`（`infer.sh` 同名）。落点三处但数学等价：训练在 `UnZipLoRALinearLayer._active_mask_content`（LoRA up 之后乘 mask）；UnZip 推理 pipeline 在 `prompt_embeds_content * rare_mask`（prompt embedding 层面置零）；Single-LoRA 推理 pipeline 在 `LoRALinearLayer._active_mask_content`（同训练机制）。`UnZipLoRALinearLayerInfer.forward` 不引用 `_active_mask_content`，所以 UnZip 推理只能走 prompt-embedding 那条路。
- **非对称秩 / 时间步秩 mask (sigma_mask)**：按 timestep 截断 LoRA 有效秩。开关 `--use_time_control`，配合 `--min_rank_content/style`、`--alpha`、`--timestep_mode`。和 TFM 完全独立，可单独开关。`use_time_control_value` 控这个，不要和 `focus_value` 混。
- **TAL (Text Alignment Loss)**：训练用的对齐损失，对应 `--with_align_loss`；和 TFM 在 `train_unziplora.py:914` 的注释里并列出现。

### 当前分支扩展（基础三策略之外）
`add_single_lora_pipeline` 分支在原版 UnZipLoRA 之上引入了若干仍在迭代的训练增强；`train.sh` 默认开启它们，改动这些路径前应该先看代码再调flag：
- **SVD初始化 + base weight 减法**（`--with_svd_init`、`--use_base_weight`、`--sig_type`、`--alpha`）：LoRA down/up 用基础权重的SVD分解初始化，推理时减回基础权重残差，保持与原始UNet等价的起点。
- **非对称秩 + 时间步mask**（`--min_rank_content`、`--min_rank_style`、`--timestep_mode={priecewise,linear}`、`--use_time_control`）：content/style LoRA 在不同时间步使用不同有效秩，由 `timestep_mode` 控制衰减形状。`priecewise` 是分段（注意拼写就是这样，代码里以此为准）。**注意：这不是 TFM**，详见上面"关键术语"。
- **对齐损失 / TAL**（`--with_align_loss`、`--align_loss_weight`，代码注释里叫 TAL）：把 rare token 的 cross-attention 对齐到 class word 的注意力分布。
- **GSA 损失**（`--with_gsa_loss`、`--gsa_loss_weight`）：用 KL/最大熵约束注意力分布的generative spatial attention正则项（最近几次commit改过实现，看代码以最新为准）。
- **Heatmap focus / TFM**（`--focus`、`--with_content_heatmap`、`--heatmap_steps/alpha/map_size`）：rare-token 文本对齐 mask，详见上面"关键术语"。`infer.sh` 通过 `--focus_value` 在推理阶段同步开关。
- **Single LoRA pipeline**（`unziplora_unet/singlelora.py`）：剥离出的单LoRA路径，用于消融或推理时对比。

### 训练增强相关参数（rare/class word）
`--content_rare_word`、`--class_word`、`--style_rare_word`、`--super_word` 同时被 align loss、gsa loss、heatmap focus 用作token索引的来源。改动这些 flag 时要保证和 `train.sh` 里 `PROMPT` / `CONTENT_FORWARD_PROMPT` / `STYLE_FORWARD_PROMPT` 中实际出现的token一致，否则注意力对齐会找不到目标token。

### 工具模块 (`unziplora_unet/utils.py`)
- `insert_unziplora_to_unet()`：推理时将训练好的content/style LoRA注入UNet
- `load_pipeline_from_sdxl()`：构建`StableDiffusionXLUnZipLoRAPipeline`
- `lora_merge_cone_select()`：cone score计算和column mask更新的核心函数
- `insert_mask()`/`generate_mask_in_unet()`：block separation的mask生成和注入

### 自定义UNet模块 (`unziplora_unet/`)
- 基于diffusers 0.25.0的UNet/Attention/Pipeline修改版，支持双hidden_states（content和style）的前向传播
- `pipeline_stable_diffusion_xl.py`：修改后的SDXL pipeline，支持`prompt_content`和`prompt_style`参数分别编码
- `attention_processor.py`/`attention.py`：attention层支持将不同prompt编码的hidden_states分别传给content/style LoRA
- `singlelora.py`：单LoRA层（非unzip），用于ablation / single-pipeline推理
- `lora.py`、`unet_2d_condition.py`、`unet_block.py`、`transformer_2d.py`：均为diffusers组件的修改版本，与上述双hidden_states前向相配

### 分析与日志
- `record_utils/cone.py`：cone score / heatmap 离线分析脚本，配合 `--with_grad_record` 产生的数据使用
- 训练日志走 wandb（`--report_to=wandb`、`WANDB_PROJECT`、`WANDB_NAME`、`--entity`），本地缓存在 `wandb/`
- 训练产物在 `models/`，验证图保存在 `output/`

### 已知遗留 issue（修之前先看一下）
追踪改动时如果碰到下面这些点，是已知 bug 不是你引入的：
- `infer.py:203` 语法错误：`content_rare_word=args.content_rare_word if args.focus_value else None,,` 末尾双逗号，整文件 `SyntaxError`，删一个逗号即可。不修这条，infer.sh 任何分支都跑不动。
- `unziplora_unet/unziplora_linear_layer.py:491` 计算 `D_style_base` 时乘的是 `merge_content * sigma_mask_content`，应为 `merge_style * sigma_mask_style`。仅在 `forward_type=="both"` + `use_base_weight=True` 时影响 style 分支减基矩阵。
- `infer.sh:78` 把 `--content_rare_word="sks"` 硬编码，与 `train.sh` 的 `CONTENT_RARE_TOKEN` 变量不同步；换数据集 / 换 rare token 时要手改两处。
- `UnZipLoRALinearLayerInfer.forward`（推理类）不引用 `_active_mask_content`，所以 UnZip 推理路径的 TFM 只能靠 prompt-embedding 那条路实现，不能复用训练里 `set_content_mask` 的接口。要统一两条路径就得在 Infer 类里补一条分支。

### 特殊标识符
- `monadikos`：训练中用于内容subject的rare token placeholder（类似DreamBooth的`sks`）
