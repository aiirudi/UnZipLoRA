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

## 核心架构

### 训练流程 (`train_unziplora.py`)
- 基于 accelerate 的训练脚本，同时优化 content 和 style 两个 LoRA
- 使用三组独立学习率：content_learning_rate、style_learning_rate、weight_learning_rate
- 训练过程中交替进行：正常LoRA训练 → cone score累积 → column mask更新 → masked训练
- 输出产物：`{output_dir}_content/`（content LoRA）、`{output_dir}_style/`（style LoRA）、`{output_dir}_merger_content.pth`、`{output_dir}_merger_style.pth`（merge权重）

### UnZipLoRA层 (`unziplora_unet/unziplora_linear_layer.py`)
- `UnZipLoRALinearLayer`：训练用，包含 content/style 两组 down/up 矩阵 + merge向量 + mask
- `UnZipLoRALinearLayerInfer`：推理用的简化版本
- 每层维护：`lora_matrix_dic`(nn.ModuleDict存储down/up权重)、`merge_content/merge_style`(可学习软mask)、`mask_content/mask_style`(硬mask，bool)、`column_score_content/column_score_style`(cone得分)
- `forward_type`控制前向传播模式："both"/"content"/"style"

### 三种分离策略
1. **Column Separation**（`with_period_column_separation`）：基于cone score选择各LoRA激活的列子集，通过`mask_updated_elements`实现
2. **Block Separation**（`with_freeze_unet`）：UNet不同block分配给不同LoRA，配置在`SDXL_content_layer_mask`/`SDXL_style_layer_mask`
3. **Weight Similarity Loss**（`similarity_lambda`）：merge向量的余弦相似度惩罚，鼓励content/style使用不同列

### 工具模块 (`unziplora_unet/utils.py`)
- `insert_unziplora_to_unet()`：推理时将训练好的content/style LoRA注入UNet
- `load_pipeline_from_sdxl()`：构建`StableDiffusionXLUnZipLoRAPipeline`
- `lora_merge_cone_select()`：cone score计算和column mask更新的核心函数
- `insert_mask()`/`generate_mask_in_unet()`：block separation的mask生成和注入

### 自定义UNet模块 (`unziplora_unet/`)
- 基于diffusers 0.25.0的UNet/Attention/Pipeline修改版，支持双hidden_states（content和style）的前向传播
- `pipeline_stable_diffusion_xl.py`：修改后的SDXL pipeline，支持`prompt_content`和`prompt_style`参数分别编码
- `attention_processor.py`/`attention.py`：attention层支持将不同prompt编码的hidden_states分别传给content/style LoRA

### 特殊标识符
- `monadikos`：训练中用于内容subject的rare token placeholder（类似DreamBooth的`sks`）
