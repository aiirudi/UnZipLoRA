#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import functools
import gc
import itertools
import logging
import math
import os
from distutils.util import strtobool
import random
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
)

from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from unziplora_unet.unziplora_linear_layer import UnZipLoRALinearLayer
from unziplora_unet.pipeline_stable_diffusion_xl import StableDiffusionXLUnZipLoRAPipeline
from unziplora_unet.unet_2d_condition import UNet2DConditionModel
from unziplora_unet.utils import *


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__)

# TODO: This function should be removed once training scripts are rewritten in PEFT
def text_encoder_lora_state_dict(text_encoder):
    state_dict = {}
    # * Get the dictionary of text encoder self attention layers                                                                                                                                                                                             
    def text_encoder_attn_modules(text_encoder):
        from transformers import CLIPTextModel, CLIPTextModelWithProjection

        attn_modules = []

        if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
            for i, layer in enumerate(text_encoder.text_model.encoder.layers):
                name = f"text_model.encoder.layers.{i}.self_attn"
                mod = layer.self_attn
                attn_modules.append((name, mod))

        return attn_modules

    for name, module in text_encoder_attn_modules(text_encoder):
        for k, v in module.q_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.q_proj.lora_linear_layer.{k}"] = v

        for k, v in module.k_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.k_proj.lora_linear_layer.{k}"] = v

        for k, v in module.v_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.v_proj.lora_linear_layer.{k}"] = v

        for k, v in module.out_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.out_proj.lora_linear_layer.{k}"] = v

    return state_dict


def save_model_card(
    repo_id: str,
    images=None,
    base_model=str,
    train_text_encoder=False,
    instance_prompt=str,
    validation_prompt=str,
    repo_folder=None,
    vae_path=None,
):
    img_str = "widget:\n" if images else ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"""
        - text: '{validation_prompt if validation_prompt else ' ' }'
          output:
            url:
                "image_{i}.png"
        """

    yaml = f"""
---
tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- text-to-image
- diffusers
- lora
- template:sd-lora
{img_str}
base_model: {base_model}
instance_prompt: {instance_prompt}
license: openrail++
---
    """

    model_card = f"""
# SDXL LoRA DreamBooth - {repo_id}

<Gallery />

## Model description

These are {repo_id} LoRA adaption weights for {base_model}.

The weights were trained  using [DreamBooth](https://dreambooth.github.io/).

LoRA for the text encoder was enabled: {train_text_encoder}.

Special VAE used for training: {vae_path}.

## Trigger words

You should use {instance_prompt} to trigger the image generation.

## Download model

Weights for this model are available in Safetensors format.

[Download]({repo_id}/tree/main) them in the Files & versions tab.

"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    #预训练模型下载位置
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    ) # 1

    # 同理 vae 路径
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    ) # 不用传，在 single lora 中不可能为 None

    # HuggingFace模型版本号 
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    ) # 1

    # 本地数据集目录
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    ) #1

    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    ) #1

    # 这个和 dataset_name 二选一， 不能两个都填入
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    ) #1

    # 从hugging face 上下载模型后的拉取地址
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    # 指定数据集中图像所在列名
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    #指定数据集中图像描述文本所在列名
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    #将训练数据重复多少次 （数据增强）
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to repeat the training data.",
    )
    #content class图片目录（不够会自动生成）
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    #content prior preservation的class prompt（如"a dog"） 
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    # style class图片目录
    parser.add_argument(
        "--class_data_dir_2",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    # style prior preservation的class prompt（如"an image in watercolor style"）
    parser.add_argument(
        "--class_prompt_2",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )

    # 训练图片的完整描述prompt（content+style）
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    ) #1
    # * Modified prompt
    # 训练时style LoRA分支接收的prompt（只描述风格）  
    parser.add_argument(
        "--style_forward_prompt",
        type=str,
        default=None,
        help="The prompt used for style during training",
    )
    # 训练时content LoRA分支接收的prompt（只描述内容）
    parser.add_argument(
        "--content_forward_prompt",
        type=str,
        default=None,
        help="The prompt used for content during training",
    ) #1

    # content验证描述（非None时开启验证）
    parser.add_argument(
        "--validation_content",
        type=str,#
        default=None,
        help="The prompt for content used during validation",
    )
    # style验证描述（非None时开启验证）
    parser.add_argument(
        "--validation_style",
        type=str,
        default=None,
        help="The prompt for style used during validation to verify that the model is learning.",
    )
    # 验证时的合并prompt（content+style一起推理）
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="The prompt used for combination during validation",
    )

    
    # 验证时只用style LoRA的prompt 
    parser.add_argument(
        "--validation_prompt_style",
        type=str,
        default=None,
        help="The prompt used for combination for style during validation.",
    )
    # 验证时只用content LoRA的prompt 
    parser.add_argument(
        "--validation_prompt_content",
        type=str,
        default=None,
        help="The prompt used for combination for content during validation.",
    )

    # 自定义扩展
    parser.add_argument(
        "--feature_prompt",
        type=str,
        default=None,
        help="Some other features we want the model to separate"
    )
    # 每次验证时生成图片数量
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    # 每隔多少个 epoch 做一次验证
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    
    # 是否启用先验保持损失，这个通常都是 DreamBoot Dataset 训练方式时启动
    # 为了避免模型在微调之后只记住微调之后的图片，但是破坏原有分布
    parser.add_argument(
        "--with_prior_preservation",
        action="store_true",
        help="Flag to add prior preservation loss.",
    )

    # content prior loss权重 
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    # style prior loss权重   
    parser.add_argument(
        "--prior_loss_weight_2",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )

    # class图片数量，不够自动用SDXL生成补齐
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )

    # 训练输出目录
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    ) #1
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    
    # 图像分辨率预处理
    parser.add_argument(
        "--resolution",
        type=int,#
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ), 
    ) # 1
    #裁剪左上角坐标嵌入，这是 SDXL 中特有的一类条件信息，用于让 UNet 知道图像 crop 的位置信息
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=(
            "Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=(
            "Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."
        ),
    )

    # 是否启用中心裁剪
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )

    # 是否训练text encoder（代码中raise NotImplementedError，实际不可用）
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    # 每轮的 batch 数目
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    ) #1
    # 采样生成图片的 batch_size
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    # 训练 epochs， 一般调 lora 训练个 600 - 1000 步这样就够了所以就默认 epochs 为 1
    parser.add_argument("--num_train_epochs", type=int, default=1)
    
    # 总训练步数（设了覆盖epoch） 
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    
    # content LoRA的down/up矩阵学习率，默认1e-4
    # * set learning rate for content lora, style lora + filter
    parser.add_argument(
        "--content_learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    ) #1
    # style LoRA的down/up矩阵学习率，默认1e-4
    parser.add_argument(
        "--style_learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )# 1
    # merge向量的学习率，默认0（阶段二才通过开启梯度生效）
    parser.add_argument(
        "--weight_learning_rate",
        type=float,
        default=0,
        help="Initial learning rate (after the potential warmup period) to use.",
    ) # 有，对应 learning_rate
    # weight similarity loss的系数，默认0.01，列分离中正交损失的系数
    parser.add_argument(
        "--similarity_lambda",
        type=float,
        default=0.01,
        help="an appropriate multiplier for the cosine similarity loss term",
    )
    # text encoder学习率（未实际使用）, 一般不会训练的， 直接加载进来就用了
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_decouple",
        type=bool,
        default=True,
        help="Use AdamW style decoupled weight decay",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for unet params",
    )
    parser.add_argument(
        "--adam_weight_decay_text_encoder",
        type=float,
        default=1e-03,
        help="Weight decay to use for text_encoder",
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    
    #自动生成class图片时的精度
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    # * Added flags or parameters
    parser.add_argument(
        "--with_finetune_mask",
        action="store_true",
        # type=lambda x: bool(strtobool(str(x))),
        # default=False,
        help="Flag to only train the overlap mask or all mask.",
    )
    parser.add_argument(
        "--with_saved_per_validation",
        action="store_true",
        # type=lambda x: bool(strtobool(str(x))),
        # default=False,
        help="Flag to store model when validation",
    )
    parser.add_argument(
        "--with_image_per_validation",
        action="store_true",
        # type=lambda x: bool(strtobool(str(x))),
        # default=False,
        help="Flag to store generated images when validation",
    )

    # action="store_false"，默认True开启block separation；传入此flag则关闭  
    parser.add_argument(
        "--with_freeze_unet",
        action="store_false",
        # default=True,
        # type=lambda x: bool(strtobool(str(x))),
        help="Flag to add block separation",
    )
    # action="store_false"，默认True开启column separation；传入此flag则关闭 
    # * column separation parameters
    parser.add_argument(
        "--with_period_column_separation",
        # type=lambda x: bool(strtobool(str(x))),
        # default=True,
        action="store_false",
        help="Flag to add columns periodically",
    )
    # column separation中cone采样轮数，默认10（训练被分成10个周期）
    parser.add_argument(
        "--sample_times",
        type=int, 
        default=10,
        help="How many time the filters are updated"
    )
    # 每轮选择的列比例，默认0.05（每轮选5%的列）
    parser.add_argument(
        "--column_ratio",
        type=float, 
        default=0.05,
        help="How much columns will be added each time"
    )
    # action="store_false"，默认True，style选列时避开content已选的列；传入则允许重叠
    parser.add_argument(
        "--with_no_overlap_first",
        action="store_false",
        # type=lambda x: bool(strtobool(str(x))),
        # default=True,
        help="Flag to add style columns that avoid overlap with subject columns",
    )
    # action="store_false"，默认True，cone score跨步累积；传入则每步独立计算
    parser.add_argument(
        "--with_accumulate_cone",
        # type=lambda x: bool(strtobool(str(x))),
        action="store_false",
        # default=True,
        help="Flag to compute the cone accumulatively (compute for one epoch)",
    )

    parser.add_argument(
        "--with_grad_record",
        action="store_true",
        help="Flag whether log the gradient change in wandb.",
    )

    parser.add_argument(
        "--with_one_shot",
        action="store_false",
        help="Flag to use one-shot training.",
    )
    # LoRA的秩，默认4
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    ) #1
    parser.add_argument("--name", type=str, default=None, help="Name of wandb project",) #有了，对应我那边的 project_name
    parser.add_argument("--tags", nargs="*", default=[], help="Tags of wandb project",)
    parser.add_argument("--entity", type=str, default="changln")
    parser.add_argument("--wandb_dir", type=str, default=None)

    # 热力图参数
    parser.add_argument("--content_rare_word", type=str,default="",help="The content rare word to be used for cross attention regularization.",)
    parser.add_argument("--class_word",type=str,default="",help="The class word to be used for cross attention regularization.",
    )    
    
    # 检查style token 生成的热力图
    parser.add_argument("--style_rare_word",type=str,default="",help="The class word to be used for cross attention regularization.",
    ) 
    parser.add_argument("--super_word",type=str,default="",help="The class word to be used for cross attention regularization.",
    ) 
    parser.add_argument("--with_content_heatmap", action="store_true",help="Log content-only token heatmap during content validation.")
    parser.add_argument("--heatmap_steps", type=int, default=100,help="Run heatmap every N global steps. 0 disables cadence gate.")
    parser.add_argument(
        "--heatmap_map_size",
        "--heatmap_size",
        dest="heatmap_map_size",
        type=int,
        default=64,
        help="Internal heatmap map size.",
    )
    parser.add_argument("--heatmap_alpha", type=float, default=0.45,help="Overlay alpha.")

    # TFM 和 TAL 参数设置
    parser.add_argument("--focus", type=str2bool, default='false', help='Flag to add prior preservation loss.')
    parser.add_argument("--with_align_loss", type=str2bool, default='false', help='启用 rare→class 对齐损失。')
    parser.add_argument("--align_loss_weight", type=float, default=1.0, help='对齐损失权重。')

    # GSA loss 参数
    parser.add_argument("--with_gsa_loss", type=str2bool, default='false', help='启用 gsa 损失。')
    parser.add_argument("--gsa_loss_weight", type=float, default=1.0, help='gsa 损失权重。')

    # 随机矩阵svd分解初始化
    parser.add_argument("--sig_type",type=str,default="last",help="随机矩阵svd分解初始化方式",
    ) 
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError(
            "Specify only one of `--dataset_name` or `--instance_data_dir`"
        )

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for content images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for content images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn(
                "You need not use --class_data_dir without --with_prior_preservation."
            )
        if args.class_prompt is not None:
            warnings.warn(
                "You need not use --class_prompt without --with_prior_preservation."
            )
        if args.class_data_dir_2 is not None:
            warnings.warn(
                "You need not use --class_data_dir_2 without --with_prior_preservation."
            )
        if args.class_prompt_2 is not None:
            warnings.warn(
                "You need not use --class_prompt_2 without --with_prior_preservation."
            )

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        class_prompt,
        device,
        class_data_root=None,
        class_prompt_2=None,
        class_data_root_2=None,
        class_num=None,
        size=1024,
        repeats=1,
        center_crop=False,
        one_shot=False,
    ):
        self.size = size
        self.center_crop = center_crop

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt
        self.custom_instance_prompts_2 = None
        self.class_prompt_2 = class_prompt_2
        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset(if the training dataset is an "cashed dataset")
        if args.dataset_name is not None:
            raise NotImplementedError
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")
            instance_images = [
                Image.open(path) for path in list(Path(instance_data_root).iterdir())
            ]
            if one_shot is True: 
                selected_image = random.choice(instance_images)
                instance_images = [selected_image]
                print(selected_image)
            self.custom_instance_prompts = None

        self.instance_images = []
        self.style_vector = []
        self.content_vector = []
        for img in instance_images:
            self.instance_images.extend(itertools.repeat(img, repeats))
        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None
        
        if class_data_root_2 is not None:
            self.class_data_root_2 = Path(class_data_root_2)
            self.class_data_root_2.mkdir(parents=True, exist_ok=True)
            self.class_images_path_2 = list(self.class_data_root_2.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root_2 = None
        
        
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.instance_images[index % self.num_instance_images]
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt

        else:  # costum prompts were provided, but length does not match size of image dataset
            example["instance_prompt"] = self.instance_prompt

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.class_prompt
        if self.class_data_root_2:
            class_image = Image.open(
                self.class_images_path_2[index % self.num_class_images]
            )
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images_2"] = self.image_transforms(class_image)
            example["class_prompt_2"] = self.class_prompt
        return example


def collate_fn(examples, with_prior_preservation=False, with_vector_loss=False, class_data_root=None, class_data_root_2=None):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values_dict = {"pixel_values_both": pixel_values}
    vector_dict = {}
    prompts_dict = {"prompts_both": prompts}
    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        if class_data_root is not None:
            pixel_values_content = [example["class_images"] for example in examples]
            prompts_content = [example["class_prompt"] for example in examples]
            pixel_values_content = torch.stack(pixel_values_content)
            pixel_values_content = pixel_values_content.to(memory_format=torch.contiguous_format).float()
            pixel_values_dict["pixel_values_content"] = pixel_values_content
            prompts_dict["prompts_content"] = prompts_content
        
        if class_data_root_2 is not None: 
            pixel_values_style = [example["class_images_2"] for example in examples]
            prompts_style = [example["class_prompt_2"] for example in examples]
            pixel_values_style = torch.stack(pixel_values_style)
            pixel_values_style = pixel_values_style.to(memory_format=torch.contiguous_format).float()
            pixel_values_dict["pixel_values_style"] = pixel_values_style
            pixel_values_dict["pixel_values_style"] = pixel_values_style
            pixel_values_dict["pixel_values_style"] = pixel_values_style
            prompts_dict["prompts_style"] = prompts_style

    batch = {
        "pixel_values_dict": pixel_values_dict, 
        "prompts_dict": prompts_dict,
        "vector_dict": vector_dict
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1) # 这里本质上相当于一种 contiguous 写法
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    def gen_prior_images(class_data_dir, class_prompt):
        # Generate class images if prior preservation is enabled.
        # Use list: if we need to insert multiple tokens
        if args.with_prior_preservation:
            if not isinstance(class_data_dir, list):
                class_data_dirs = [class_data_dir]
            for class_data_dir in class_data_dirs:
                if class_data_dir is not None:
                    class_images_dir = Path(class_data_dir)
                    if not class_images_dir.exists():
                        class_images_dir.mkdir(parents=True)
                    cur_class_images = len(list(class_images_dir.iterdir()))

                    if cur_class_images < args.num_class_images:
                        torch_dtype = (
                            torch.float16 if accelerator.device.type == "cuda" else torch.float32
                        )
                        if args.prior_generation_precision == "fp32":
                            torch_dtype = torch.float32
                        elif args.prior_generation_precision == "fp16":
                            torch_dtype = torch.float16
                        elif args.prior_generation_precision == "bf16":
                            torch_dtype = torch.bfloat16
                        pipeline = StableDiffusionXLPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            torch_dtype=torch_dtype,
                            revision=args.revision,
                        )
                        pipeline.set_progress_bar_config(disable=True)

                        num_new_images = args.num_class_images - cur_class_images
                        logger.info(f"Number of class images to sample: {num_new_images}.")

                        sample_dataset = PromptDataset(class_prompt, num_new_images)
                        sample_dataloader = torch.utils.data.DataLoader(
                            sample_dataset, batch_size=args.sample_batch_size
                        )

                        sample_dataloader = accelerator.prepare(sample_dataloader)
                        pipeline.to(accelerator.device)

                        for example in tqdm(
                            sample_dataloader,
                            desc="Generating class images",
                            disable=not accelerator.is_local_main_process,
                        ):
                            images = pipeline(example["prompt"]).images

                            for i, image in enumerate(images):
                                hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                                image_filename = (
                                    class_images_dir
                                    / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                                )
                                image.save(image_filename)

                        del pipeline
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
    
    gen_prior_images(args.class_data_dir, args.class_prompt)
    gen_prior_images(args.class_data_dir_2, args.class_prompt_2)
    
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    # 这里其实就是导入两个 class 一个是 CLIPTextEncoder， 一个是 CLIPTextEncoderWithProjection
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)
    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)

    # The VAE is always in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32)
    
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # xformers 启用内存高效注意力机制
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                    "please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    # now we will add new LoRA weights to the attention layers
    # Set correct lora layers
    unet_lora_parameters_content = []
    unet_lora_parameters_style = []
    unet_lora_weight = []
    
    # 遍历模型并把模型中的 Attention blocks 的 q,k,v 全部设置 LoRA layer
    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # Parse the attention module.
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)

        # TFM 的 mask 只在 cross-attention (attn2) 的 to_k / to_v 上启用
        is_cross = ".attn2." in attn_processor_name

        # Set the `lora_layer` attribute of the attention-related matrices.
        attn_module.to_q.set_lora_layer(
            UnZipLoRALinearLayer(
                in_features=attn_module.to_q.in_features,
                out_features=attn_module.to_q.out_features,
                rank=args.rank,
                lora_matrix_num = 2,
                device=accelerator.device,
                # dtype=weight_dtype,
                lora_matrix_key = ["content", "style"],
                use_mask=False,
                sig_type=args.sig_type
            )
        )
        attn_module.to_k.set_lora_layer(
            UnZipLoRALinearLayer(
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
                rank=args.rank,
                lora_matrix_num = 2,
                device=accelerator.device,
                # dtype=weight_dtype,
                lora_matrix_key = ["content", "style"],
                use_mask=is_cross,
                sig_type=args.sig_type
            )
        )
        attn_module.to_v.set_lora_layer(
            UnZipLoRALinearLayer(
                in_features=attn_module.to_v.in_features,
                out_features=attn_module.to_v.out_features,
                rank=args.rank,
                lora_matrix_num = 2,
                device=accelerator.device,
                # dtype=weight_dtype,
                lora_matrix_key = ["content", "style"],
                use_mask=is_cross,
                sig_type=args.sig_type
            )
        )
        attn_module.to_out[0].set_lora_layer(
            UnZipLoRALinearLayer(
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                rank=args.rank,
                lora_matrix_num = 2,
                device=accelerator.device,
                # dtype=weight_dtype,
                lora_matrix_key = ["content", "style"],
                use_mask=False,
                sig_type=args.sig_type
            )
        )


        # TODO: Only for content and style
        def collect_params(lora_module, parameters, model_key=None):
            for key, layer in lora_module.lora_matrix_dic.items():
                if model_key in key:
                    parameters.extend(layer.parameters())
            return parameters
        
        def collect_weight(lora_module, parameters, model_key=None):
            weight = getattr(lora_module, f"merge_{model_key}")
            weight.requires_grad = False 
            parameters.append(weight)
            return parameters

        # 分开统计 content_lora 和 style_lora 的参数
        unet_lora_parameters_content = collect_params(attn_module.to_q.lora_layer, unet_lora_parameters_content, model_key="content")
        unet_lora_parameters_content = collect_params(attn_module.to_k.lora_layer, unet_lora_parameters_content, model_key="content")
        unet_lora_parameters_content = collect_params(attn_module.to_v.lora_layer, unet_lora_parameters_content, model_key="content")
        unet_lora_parameters_content = collect_params(attn_module.to_out[0].lora_layer, unet_lora_parameters_content, model_key="content")
        unet_lora_parameters_style = collect_params(attn_module.to_q.lora_layer, unet_lora_parameters_style, model_key="style")
        unet_lora_parameters_style = collect_params(attn_module.to_k.lora_layer, unet_lora_parameters_style, model_key="style")
        unet_lora_parameters_style = collect_params(attn_module.to_v.lora_layer, unet_lora_parameters_style, model_key="style")
        unet_lora_parameters_style = collect_params(attn_module.to_out[0].lora_layer, unet_lora_parameters_style, model_key="style")

        # 统计全部LoRA layer 的参数
        unet_lora_weight = collect_weight(attn_module.to_q.lora_layer, unet_lora_weight, model_key="content")
        unet_lora_weight = collect_weight(attn_module.to_k.lora_layer, unet_lora_weight, model_key="content")
        unet_lora_weight = collect_weight(attn_module.to_v.lora_layer, unet_lora_weight, model_key="content")
        unet_lora_weight = collect_weight(attn_module.to_out[0].lora_layer, unet_lora_weight, model_key="content")
        unet_lora_weight = collect_weight(attn_module.to_q.lora_layer, unet_lora_weight, model_key="style")
        unet_lora_weight = collect_weight(attn_module.to_k.lora_layer, unet_lora_weight, model_key="style")
        unet_lora_weight = collect_weight(attn_module.to_v.lora_layer, unet_lora_weight, model_key="style")
        unet_lora_weight = collect_weight(attn_module.to_out[0].lora_layer, unet_lora_weight, model_key="style")

    if args.seed is not None:
        set_seed(args.seed)
    
    # print(unet_lora_parameters)
    # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks.
    if args.train_text_encoder:
        raise NotImplementedError
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        text_lora_parameters_one = LoraLoaderMixin._modify_text_encoder(
            text_encoder_one, dtype=torch.float32, rank=args.rank
        )
        text_lora_parameters_two = LoraLoaderMixin._modify_text_encoder(
            text_encoder_two, dtype=torch.float32, rank=args.rank
        )
    
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    # Save the model in the formate that the lora weight can be loaded by diffuser later
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_lora_layers_to_save_content, unet_lora_layers_merger_content, unet_lora_layers_base_weight_content = unet_inverse_ziplora_state_dict(model, key="content")
                    unet_lora_layers_to_save_style, unet_lora_layers_merger_style, unet_lora_layers_base_weight_style = unet_inverse_ziplora_state_dict(model, key="style")
                elif isinstance(
                    model, type(accelerator.unwrap_model(text_encoder_one))
                ):
                    text_encoder_one_lora_layers_to_save = text_encoder_lora_state_dict(
                        model
                    )
                elif isinstance(
                    model, type(accelerator.unwrap_model(text_encoder_two))
                ):
                    text_encoder_two_lora_layers_to_save = text_encoder_lora_state_dict(
                        model
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=f"{args.output_dir}_content",
                unet_lora_layers=unet_lora_layers_to_save_content,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=f"{args.output_dir}_style",
                unet_lora_layers=unet_lora_layers_to_save_style,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )
    
    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(
            lora_state_dict, network_alphas=network_alphas, unet=unet_
        )

        text_encoder_state_dict = {
            k: v for k, v in lora_state_dict.items() if "text_encoder." in k
        }
        LoraLoaderMixin.load_lora_into_text_encoder(
            text_encoder_state_dict,
            network_alphas=network_alphas,
            text_encoder=text_encoder_one_,
        )

        text_encoder_2_state_dict = {
            k: v for k, v in lora_state_dict.items() if "text_encoder_2." in k
        }
        LoraLoaderMixin.load_lora_into_text_encoder(
            text_encoder_2_state_dict,
            network_alphas=network_alphas,
            text_encoder=text_encoder_two_,
        )

    accelerator.register_save_state_pre_hook(save_model_hook)
    # accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.content_learning_rate = (
            args.content_learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )
        args.style_learning_rate = (
            args.style_learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )
        args.weight_learning_rate = (
            args.weight_learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )
    # Optimization parameters: set learning rate for content, style and weight
    # ! Do not optimize if one branch is first trained
    params_lr_to_optimize = []
    unet_params_to_optimze = []
    unet_content_lora_parameters_with_lr = {
        "params": unet_lora_parameters_content,
        "lr": args.content_learning_rate,
    }
    unet_style_lora_parameters_with_lr = {
        "params": unet_lora_parameters_style,
        "lr": args.style_learning_rate,
    }
    unet_weight_with_lr = {
        "params": unet_lora_weight,
        "lr": args.weight_learning_rate
    }

    if args.train_text_encoder:
        raise NotImplementedError
        # different learning rate for text encoder and unet
        text_lora_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        text_lora_parameters_two_with_lr = {
            "params": text_lora_parameters_two,
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_lr_to_optimize = [
            unet_content_lora_parameters_with_lr,
            unet_style_lora_parameters_with_lr,
            text_lora_parameters_one_with_lr,
            text_lora_parameters_two_with_lr,
        ]
    else:
        params_lr_to_optimize = [unet_content_lora_parameters_with_lr, unet_style_lora_parameters_with_lr, unet_weight_with_lr]
        unet_params_to_optimze = unet_lora_parameters_content + unet_lora_parameters_style + unet_lora_weight
    
    # block seperateion 用来定义哪些快事用来训练 content， 哪些块使用来训练 style 的
    if args.with_freeze_unet: 
        mask_dictionary_content = {
            "mid_block": ["N_0_A_A"],
            "up_blocks.": ['1_A_A_A','0_1_A_A'],
            # "up_blocks.": ['1_A_A_A','0_1,2_A_A'],
            "down_blocks.": ["A_A_A_A"]
        }
        mask_dictionary_style = {
            "mid_block": ["N_0_A_A"],
            "up_blocks.": ['0_0,2_A_A'],
            # "up_blocks.": ['1_A_A_A',],
            "down_blocks.": ["A_A_A_A"]
        }
    else:
        mask_dictionary_content = {}
        mask_dictionary_style = {}
        
    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warn(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warn(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_lr_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        print("optimizer prodigy")
        try:
            import prodigyopt
        except ImportError:
            raise ImportError(
                "To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`"
            )

        optimizer_class = prodigyopt.Prodigy

        if args.content_learning_rate <= 0.1:
            logger.warn(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warn(
                f"Learning rates were provided both for the unet and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.content_learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one and text_encoder_parameters_two to be
            # --learning_rate
            params_lr_to_optimize[2]["lr"] = args.learning_rate
            params_lr_to_optimize[3]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_lr_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )
    # instance_prompt = f"{args.content_forward_prompt.strip()} {args.style_forward_prompt.strip()}"
    instance_prompt = args.instance_prompt
    
    # 这个 DreamBootDataset 实际上只会返回一个 instance image 以及 instance promopt
    # 对于 content_prompt 以及 style prompt 完全没有
    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=instance_prompt,
        class_prompt=args.class_prompt,
        device=accelerator.device,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt_2=args.class_prompt_2,
        class_data_root_2=args.class_data_dir_2 if args.with_prior_preservation else None,
        class_num=args.num_class_images,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
        one_shot=args.with_one_shot, 
    )


    # collate_fn 就是把参数中的 batch_size 作为输入，作为 collate_fn 的匿名函数的输入也就是 examples
    # 然后把这个 examples 传入自己实现的函数，来规定返回的数据格式是怎么样的
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation, args.class_data_dir, args.class_data_dir_2),
        num_workers=args.dataloader_num_workers,
    )
    # Computes additional embeddings/ids required by the SDXL UNet.
    # regular text embeddings (when `train_text_encoder` is not True)
    # pooled text embeddings
    # time ids

    def compute_time_ids():
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        original_size = (args.resolution, args.resolution)
        target_size = (args.resolution, args.resolution)
        crops_coords_top_left = (
            args.crops_coords_top_left_h,
            args.crops_coords_top_left_w,
        )
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    # 判断是否会训练 text_encoder
    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders, tokenizers, prompt
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds

    # Handle instance prompt.
    instance_time_ids = compute_time_ids() # (1, 6)
 
    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        #instance_promppt_hidden_states: (B, seq_len, 768 + 1280)
        #instance_pooles_prompt_embeds: (B, seq_len, 1280)
        (
            instance_prompt_hidden_states,
            instance_pooled_prompt_embeds,
        ) = compute_text_embeddings(instance_prompt, text_encoders, tokenizers)

    # Handle class prompt for prior-preservation.
    # 这里其实是 DreamBooth 的训练方式，DreamBooth 的核心思想是：不想让模型微调之后生成的和微调图像类别相同的content时全都和微调图像一样
    # 所以带 with_prior_preservation 这个参数我都可以不管
    if args.with_prior_preservation:
        class_time_ids = compute_time_ids()
        if not args.train_text_encoder:
            if args.class_data_dir is not None:
                (
                    class_prompt_hidden_states,
                    class_pooled_prompt_embeds,
                ) = compute_text_embeddings(args.class_prompt, text_encoders, tokenizers)
            if args.class_data_dir_2 is not None:
                (
                    class_prompt_hidden_states_2,
                    class_pooled_prompt_embeds_2,
                ) = compute_text_embeddings(args.class_prompt_2, text_encoders, tokenizers)
    
    # 生成 style_prompt_hidden_states , 以及要加入 time_embedding 的 style_pooled_prompt_hidden_state
    (
        style_forward_prompt_hidden_states,
        style_forward_pooled_prompt_embeds,
    ) = compute_text_embeddings(args.style_forward_prompt, text_encoders, tokenizers)

    # 生成 content_prompt_hidden_states , 以及要加入 time_embedding 的 content_pooled_prompt_hidden_state
    (
        content_forward_prompt_hidden_states,
        content_forward_pooled_prompt_embeds,
    ) = compute_text_embeddings(args.content_forward_prompt, text_encoders, tokenizers)


    # Clear the memory here
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        del tokenizers, text_encoders
        gc.collect()
        torch.cuda.empty_cache()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.
    add_time_ids = instance_time_ids
    
    if not train_dataset.custom_instance_prompts:
        if not args.train_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            unet_add_text_embeds = instance_pooled_prompt_embeds
            prompt_embeds_style_forward = style_forward_prompt_hidden_states
            unet_add_text_embeds_style_forward = style_forward_pooled_prompt_embeds
            prompt_embeds_content_forward = content_forward_prompt_hidden_states
            unet_add_text_embeds_content_forward = content_forward_pooled_prompt_embeds
            
            if args.with_prior_preservation:
                if args.class_data_dir is not None:
                    prompt_embeds_content = class_prompt_hidden_states
                    unet_add_text_embeds_content = class_pooled_prompt_embeds
                if args.class_data_dir_2 is not None:
                    prompt_embeds_style = class_prompt_hidden_states_2
                    unet_add_text_embeds_style = class_pooled_prompt_embeds_2

        # if we're optmizing the text encoder (both if instance prompt is used for all images or custom prompts) we need to tokenize and encode the
        # batch prompts on all training steps
        else:
            tokens_one = tokenize_prompt(tokenizer_one, instance_prompt)
            tokens_two = tokenize_prompt(tokenizer_two, instance_prompt)
            
            if args.with_prior_preservation:
                class_tokens_one = tokenize_prompt(tokenizer_one, args.class_prompt)
                class_tokens_two = tokenize_prompt(tokenizer_two, args.class_prompt)
                class_tokens_one_2 = tokenize_prompt(tokenizer_one, args.class_prompt_2)
                class_tokens_two_2 = tokenize_prompt(tokenizer_two, args.class_prompt_2)
                tokens_one_content = class_tokens_one, class_tokens_one_2
                tokens_two_content = class_tokens_two, class_tokens_two_2
                tokens_one_style = class_tokens_one_2
                tokens_two_style = class_tokens_two_2

    # 这里卡住了，我最后算出来是只用训练一步了
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    # 这段代码就是将每个对象移动到适合它们的设备上进行分布式训练
    if args.train_text_encoder:
        (
            unet,
            text_encoder_one,
            text_encoder_two,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            unet,
            text_encoder_one,
            text_encoder_two,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # * When start training, doesn't allow to add orthognal loss ==> We want to know the importance of each column without coefficient
    with_orthognal = False
    loss_orthognal = None
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    sampled_steps = math.ceil(args.max_train_steps / args.sample_times)
    
    print(f"Default values: gradient_accumulation_steps {args.gradient_accumulation_steps}, num_train_epochs {args.num_train_epochs}, steps per epoch {num_update_steps_per_epoch}")
    print(f"Computed value: len of dataset {len(train_dataset)} len of dataloader {len(train_dataloader)}, max_train_steps {args.max_train_steps}")
    print(f"Sampled steps {sampled_steps}")
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process: # 这段代码的主要作用就是初始化 WandB 跟踪器
        config = vars(args)
        if args.wandb_dir is None:
            accelerator.init_trackers("dreambooth-inzerse_ziplora_rank-sd-xl", config=config, 
                init_kwargs={
                    "wandb": {
                    "entity": args.entity
                    }
                })
        else:
            accelerator.init_trackers("dreambooth-inzerse_ziplora_rank-sd-xl", config=config, 
                init_kwargs={
                    "wandb": {
                    "entity": args.entity,
                    "dir": args.wandb_dir
                    }
                })
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                tags = []
                for i in args.tags:
                    tags.extend(i.split(","))
                tags = [tag.strip() for tag in tags]
                wandb.run.tags = tags
    # 验证函数，在训练过程中生成样本图片来监控训练效果
    def log_validation(
        pipeline,
        args,
        accelerator,
        validation_prompt=None,
        validation_prompt_content=None,
        validation_prompt_style=None,
        generate_heatmap=False,
        tokenizer_one=None, 
        tokenizer_two=None,
        heatmap_type='content'
    ):
        logger.info(
            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
            f" {validation_prompt}."
        )
        scheduler_args = {}

        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        
        # run inference
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
        # Currently the context determination is a bit hand-wavy. We can improve it in the future if there's a better
        # way to condition it. Reference: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
        if pipeline.__class__.__name__ == 'StableDiffusionXLUnZipLoRAPipeline':
            pipeline_args = {"prompt": validation_prompt, 
                            "prompt_content": validation_prompt_content, 
                            "prompt_style": validation_prompt_style}
        else:   
            pipeline_args = {"prompt": validation_prompt}
        
        if not generate_heatmap:
            images = [pipeline(**pipeline_args, generator=generator, negative_prompt=", ".join(f"({w}:1.2)" for w in universal_nevigate)).images[0] for _ in range(args.num_validation_images)]
            heatmap_panels = []
        else:
            images, heatmap_panels = [], []

            recs, original_procs = None, None

            if heatmap_type == 'content':
                rare1, cls1 = build_token_masks(tokenizer_one, [validation_prompt], args.content_rare_word, args.class_word, accelerator.device)
                rare2, cls2 = build_token_masks(tokenizer_two, [validation_prompt], args.content_rare_word, args.class_word, accelerator.device)
            else:
                rare1, cls1 = build_token_masks(tokenizer_one, [validation_prompt], args.style_rare_word, args.super_word, accelerator.device)
                rare2, cls2 = build_token_masks(tokenizer_two, [validation_prompt], args.style_rare_word, args.super_word, accelerator.device)

            rare_ids = torch.where((rare1 | rare2)[0])[0].tolist()
            cls_ids = torch.where((cls1 | cls2)[0])[0].tolist()
            target_ids = sorted(set(rare_ids + cls_ids))

            recs, original_procs = attach_recorders_to_unet(pipeline.unet, target_ids, map_size=args.heatmap_map_size)

            try:
                for i in range(args.num_validation_images):
                    if recs is not None:
                        for r in recs:
                            r.reset()

                    gen = torch.Generator(device=accelerator.device).manual_seed((args.seed or 0) + i)                               
                    img = pipeline(**pipeline_args, generator=gen,negative_prompt=", ".join(f"({w}:1.2)" for w in universal_nevigate)).images[0]
                    images.append(img)  
                    if recs is not None:
                        rare_map = aggregate_maps(recs, rare_ids)
                        cls_map  = aggregate_maps(recs, cls_ids)                                                                     
                        rare_ov = overlay_heatmap(img, rare_map, args.heatmap_alpha) if rare_map is not None else img
                        cls_ov  = overlay_heatmap(img, cls_map,  args.heatmap_alpha) if cls_map  is not None else img  
                        heatmap_panels.append((rare_ov, cls_ov))              
            finally:
                for name, proc in original_procs.items():
                    module = pipeline.unet.get_submodule(name)
                    module.processor = proc 

        if accelerator.trackers[0].name == "tensorboard":
            image_lst = np.stack([np.asarray(img) for img in images])
        if accelerator.trackers[0].name == "wandb":
            image_lst = [wandb.Image(image, caption=f"{i}: {validation_prompt}") for i, image in enumerate(images)]
        
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return images, image_lst, heatmap_panels

    # 将图片列表水平拼接成一张大图
    def concatenate_horizontal_img(img_lst):
        img_lst = [img.convert('RGB') if img.mode != 'RGB' else img for img in img_lst]
        widths, heights = zip(*(img.size for img in img_lst))
        total_width = sum(widths)
        max_height = max(heights)
        # Create an empty canvas with the calculated dimensions
        concatenated_img = Image.new('RGB', (total_width, max_height))
        # Paste images next to each other
        x_offset = 0
        for img in img_lst:
            concatenated_img.paste(img, (x_offset, 0))
            x_offset += img.width
        return concatenated_img
    
    # 将图片列表竖直拼接成一张大图
    def concatenate_vertical_sublst(sublst_img):
        widths, heights = zip(*(img.size for img in sublst_img))
        max_width = max(widths)
        total_height = sum(heights)
        # Create an empty canvas with the calculated dimensions
        concatenated_img = Image.new('RGB', (max_width, total_height))
        # Paste images next to each other
        y_offset = 0
        for img in sublst_img:
            concatenated_img.paste(img, (0, y_offset))
            y_offset += img.height
        return concatenated_img
    
    
    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        raise NotImplementedError

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    # ----------------------------------------------------------------------
    # 设置捕获 LoRA 输入输出, 设置钩子函数
    ENABLE_LORA_CAPTURE = True              
    ENABLE_BASE_CAPTURE = True    
    ENABLE_STYLE_CAPTURE = True

    lora_content_capture = {"k": [], "v": []}
    base_capture = {"k": []}
    
    lora_style_probs_capture = [] # 用于捕获 style lora 的输出

    def _make_content_lora_hook(kind: str):
        def _hook(module, args, kwargs, output):
            if not ENABLE_LORA_CAPTURE:
                return 
            content_inp = args[0]

            D = module.lora_matrix_dic["content_down"].weight.T * module.merge_content
            U = module.lora_matrix_dic["content_up"].weight.T

            D_base = module.base_lora_matrix_dic["content_down"].T * module.merge_content
            U_base = module.base_lora_matrix_dic["content_up"].T

            if module.masked_matrix["content"]:
                D = D *  module.mask_content
                D_base = D_base * module.mask_content
            
            weight_content = D @ U
            base_weight_content = D_base @ U_base

            content_out = content_inp @ (weight_content - base_weight_content)
            lora_content_capture[kind].append(content_out.to(content_inp.dtype))
        return _hook

    # 在 both prompt 中加入提示词
    style_rare_1, _ = build_token_masks(tokenizer_one, [args.instance_prompt], 
    rare_word = args.style_rare_word, super_word="", device=accelerator.device)
    style_rare_2, _ = build_token_masks(tokenizer_two, [args.instance_prompt], 
    rare_word = args.style_rare_word, super_word="", device=accelerator.device)
    style_rare_mask = style_rare_1 | style_rare_2
    style_token_ids = torch.where(style_rare_mask[0])[0].tolist()
    assert len(style_token_ids) > 0, \
      f"style_rare_word='{args.style_rare_word}' 未在 instance_prompt 中找到，检查 train.sh"


    def _compute_to_k_no_hook(attn, enc, enc_c, enc_s):
        """
        等价于直接用 attn.to_k 调用forward 但是为了避免再调用一个 content LoRA 的 hook 所以这里改成直接计算
        """
        base = F.linear(enc, attn.to_k.weight, getattr(attn.to_k, "bias", None))
        
        lora = attn.to_k.lora_layer
        if lora is None or not isinstance(lora, UnZipLoRALinearLayer):
            return base
        
        dtype = lora.dtype
        hs_c = enc_c if enc_c is not None else enc
        hs_s = enc_s if enc_s is not None else hs_c
        orig_dtype = hs_c.dtype

        with torch.no_grad():
            D_c = lora.lora_matrix_dic["content_down"].weight.T * lora.merge_content
            U_c = lora.lora_matrix_dic["content_up"].weight.T

            D_c_base = lora.base_lora_matrix_dic["content_down"].T * lora.merge_content
            U_c_base = lora.base_lora_matrix_dic["content_up"].T
            if lora.masked_matrix.get("content", False):
                D_c = D_c * lora.mask_content
                D_c_base = D_c_base * lora.mask_content
            weight_content = D_c @ U_c
            base_weight_content = D_c_base @ U_c_base

            delta_c = hs_c.to(dtype) @ (weight_content - base_weight_content)

            if lora.use_mask and UnZipLoRALinearLayer._active_mask_content is not None:
                delta_c = delta_c * UnZipLoRALinearLayer._active_mask_content.to(delta_c.dtype)
        
        D_s = lora.lora_matrix_dic["style_down"].weight.T * lora.merge_style
        U_s = lora.lora_matrix_dic["style_up"].weight.T

        D_s_base = lora.base_lora_matrix_dic["style_down"].T * lora.merge_style
        U_s_base = lora.base_lora_matrix_dic["style_up"].T        

        if lora.masked_matrix.get("style", False):
            D_s = D_s * lora.mask_style
            D_s_base = D_s_base * lora.mask_style
        weight_style = D_s @ U_s
        base_weight_style = D_s_base @ U_s_base

        delta_s = hs_s.to(dtype) @ (weight_style - base_weight_style)

        ft = lora.forward_type
        if ft == 'content':
            return base + delta_c.to(orig_dtype)
        if ft == 'style':
            return base + delta_s.to(orig_dtype)
        # both
        return base + delta_c.to(orig_dtype) + delta_s.to(orig_dtype)

    def _make_style_attn_hook(style_token_ids):
        style_ids = torch.tensor(sorted(set(style_token_ids)), device=accelerator.device)
        def _hook(attn, args, kwargs, output):
            if not ENABLE_STYLE_CAPTURE:
                return 
            
            hidden_states = args[0] if len(args) > 0 else kwargs['hidden_states']
            
            enc = kwargs.get("encoder_hidden_states", None)
            enc_c = kwargs.get("encoder_hidden_states_content", None)
            enc_s = kwargs.get("encoder_hidden_states_style", None)

            if enc is None:
                return

            with torch.no_grad():
                if isinstance(attn.to_q.lora_layer, UnZipLoRALinearLayer):
                    q = attn.to_q(hidden_states,
                                hidden_states_1=hidden_states,
                                hidden_states_2=hidden_states)
                else:
                    q = attn.to_q(hidden_states)

            k = _compute_to_k_no_hook(attn, enc, enc_c, enc_s) 
            
            q = attn.head_to_batch_dim(q)
            k = attn.head_to_batch_dim(k)

            probs = attn.get_attention_scores(q, k, None)
            s = probs[..., style_ids.to(probs.device)].sum(dim=-1)
            lora_style_probs_capture.append(s)
        return _hook

    def _make_base_hook(kind: str):
        def _hook(module, args, kwargs, output):
            if not ENABLE_BASE_CAPTURE:
                return 
            x = args[0]
            if x.dim() == 3:
                b, t, d = x.shape
                base =  F.linear(x.view(-1 , d), module.weight, module.bias).view(b, t, -1)
            else:
                base = F.linear(x, module.weight, module.bias)
            base_capture[kind].append(base)
        return _hook

    # 想 LoRA Layer 中注册钩子函数
    for _name, _mod in unet.named_modules():
        if not _name.endswith(".attn2"):
            continue
            
        for _layer in [_mod.to_k, _mod.to_v]:
            if getattr(_layer, "lora_layer", None) is None:
                continue
            kind = "k" if _layer is _mod.to_k else "v"
            _layer.lora_layer.register_forward_hook(_make_content_lora_hook(kind), with_kwargs=True)
        
        # 为 style lora 加入全局对齐
        _mod.register_forward_hook(_make_style_attn_hook(style_token_ids),with_kwargs=True)
        
        _mod.to_k.register_forward_hook(_make_base_hook("k"), with_kwargs=True)
    # -------------------------------------------------------------------------
        

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()

            # set top parameter requires_grad = True for gradient checkpointing works
            text_encoder_one.text_model.embeddings.requires_grad_(True)
            text_encoder_two.text_model.embeddings.requires_grad_(True)

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                model_inputs = {}
                pixel_values_dict, prompts_dict = batch["pixel_values_dict"], batch["prompts_dict"]
                
                #打印调试查看输出
                #print(f'prompts_dict={prompts_dict}')
                
                model_inputs_key_list = ["both"]
                
                if args.with_prior_preservation:
                    if args.class_data_dir is not None:
                        model_inputs_key_list.append("content")
                    if args.class_data_dir_2 is not None:
                        model_inputs_key_list.append("style")
                
                # -----------------------------计算content_prompt的mask -----------------------------
                rare_mask_1, sueper_mask_1 = build_token_masks(
                    tokenizer_one,                              
                    args.content_forward_prompt,
                    rare_word = args.content_rare_word,                
                    super_word = args.class_word,             
                    device = accelerator.device,
                )   

                rare_mask_2, sueper_mask_2 = build_token_masks(
                    tokenizer_two,                              
                    args.content_forward_prompt,
                    rare_word = args.content_rare_word,                
                    super_word = args.class_word,             
                    device = accelerator.device,
                ) 

                rare_mask = rare_mask_1 | rare_mask_2
                super_mask = sueper_mask_1 | sueper_mask_2

                non_rare_mask = ~rare_mask
                if args.focus:
                    UnZipLoRALinearLayer.set_content_mask(rare_mask.unsqueeze(-1))
                else:
                    UnZipLoRALinearLayer.set_content_mask(None)
                
                # -----------------------------计算content_prompt的mask -----------------------------

                
                for key in model_inputs_key_list:
                    pixel_values = pixel_values_dict[f"pixel_values_{key}"].to(dtype=vae.dtype)

                    # Convert images to latent space
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = model_input * vae.config.scaling_factor
                    if args.pretrained_vae_model_name_or_path is None:
                        model_input = model_input.to(weight_dtype)
                    model_inputs[key] = model_input
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_inputs["both"])
                bsz = model_inputs["both"].shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=model_input.device,
                )
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(
                    model_inputs["both"], noise, timesteps
                )

                # 这里就是同时输入 3 个prompt： both_prompt，content_prompt, style_prompt
                # Predict the noise residual
                if not args.train_text_encoder:
                    unet_added_conditions = {
                        "time_ids": add_time_ids.repeat(bsz, 1),
                        "text_embeds": unet_add_text_embeds.repeat(
                            bsz, 1
                        ),
                    }
                    prompt_embeds_input = prompt_embeds.repeat(
                        bsz, 1, 1
                    )
                    prompt_embeds_style_forward_input = prompt_embeds_style_forward.repeat(
                        bsz, 1, 1
                    )
                    prompt_embeds_content_forward_input = prompt_embeds_content_forward.repeat(
                        bsz, 1, 1
                    )

                    # ----------------------------------------------------
                    # 每步 forward 之前先清空一次
                    for _lst in lora_content_capture.values():
                        _lst.clear()
                    for _lst in base_capture.values():
                        _lst.clear()
                    lora_style_probs_capture.clear()
                    # ----------------------------------------------------

                    # print(noisy_model_input.shape, timesteps.shape, prompt_embeds_input.shape, unet_add_text_embeds.shape)
                    model_pred = unet(
                        noisy_model_input,
                        timesteps,
                        prompt_embeds_input,
                        prompt_embeds_content_forward_input,
                        prompt_embeds_style_forward_input,
                        added_cond_kwargs=unet_added_conditions,
                    ).sample
                else:
                    raise NotImplementedError
                # 重建损失
                reconstruction_loss = F.mse_loss(
                    model_pred.float(), noise.float(), reduction="none"
                ).mean()
                loss = reconstruction_loss.clone()
                
                
                # print("all", noise.shape, model_inputs[0].shape, model_pred.shape)
                if with_orthognal:
                    # 这里计算 ziplora 中的正交损失
                    loss_orthognal = args.similarity_lambda * inverse_ziplora_compute_weight_similarity(
                        unet
                    )
                    loss += loss_orthognal
                    print("loss", loss, "orthognal", loss_orthognal)
                
                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    if args.class_data_dir is not None:
                        noise = torch.randn_like(model_inputs["content"])
                        unet_added_conditions_content = {
                        "time_ids": class_time_ids.repeat(bsz, 1),
                        "text_embeds": unet_add_text_embeds_content.repeat(bsz, 1),
                        }
                        prompt_embeds_content_input = prompt_embeds_content.repeat(bsz, 1, 1)
                        model_inputs_content = noise_scheduler.add_noise(
                            model_inputs["content"], noise, timesteps
                        )
                        unziplora_set_forward_type(unet, type="content")
                        model_pred_content = unet(
                            model_inputs_content,
                            timesteps,
                            prompt_embeds_content_input,
                            added_cond_kwargs=unet_added_conditions_content,
                        ).sample
                        prior_loss_content = F.mse_loss(
                            model_pred_content.float(), noise.float(), reduction="mean"
                        )
                        
                        unziplora_set_forward_type(unet, type="both")
                        loss = loss + args.prior_loss_weight * prior_loss_content
                    # print("content", noise.shape, model_inputs[1].shape, model_pred_content.shape)
                    if args.class_data_dir_2 is not None:
                        noise = torch.randn_like(model_inputs["style"])
                        unet_added_conditions_style = {
                        "time_ids": class_time_ids.repeat(bsz, 1),
                        "text_embeds": unet_add_text_embeds_style.repeat(bsz, 1),
                        }
                        prompt_embeds_style_input = prompt_embeds_style.repeat(bsz, 1, 1)
                        model_inputs_style = noise_scheduler.add_noise(
                            model_inputs["style"], noise, timesteps
                        )
                        unziplora_set_forward_type(unet, type="style")
                        model_pred_style = unet(
                            model_inputs_style,
                            timesteps,
                            prompt_embeds_style_input,
                            added_cond_kwargs=unet_added_conditions_style,
                        ).sample
                        # Compute prior loss
                        prior_loss_style = F.mse_loss(
                            model_pred_style.float(), noise.float(), reduction="mean"
                        ) 
                        unziplora_set_forward_type(unet, type="both")
                        loss = loss + args.prior_loss_weight_2 * prior_loss_style
                    
                
                # -----------------------------------------------------------
                """
                # 这里基本上就是这两个大小
                rare_mask.shape == super_mask.shape == non_rare_mask.shape = [B_mask, T] = [1, 77]
                base_capture 中每个项的形状都是 (bsz, 77, d_layer)
                lora_content_lora 中每个项的形状都是 (bsz, 77, d_layer)
                """

                lora_content_loss = None
                # 插入 TAL 计算 和 non-rare token 上的增量
                if args.focus and (lora_content_capture["k"] or lora_content_capture["v"]):
                    lora_content_loss_sum, n_terms = 0.0, 0
                    #mask.shape: (1, 77, 1) 
                    mask = non_rare_mask.unsqueeze(-1)

                    for delta in itertools.chain(lora_content_capture["k"], lora_content_capture["v"]):
                        delta = delta * mask
                        delta_loss = delta.abs().mean()
                        lora_content_loss_sum += delta_loss
                        n_terms += 1
                        del delta_loss, delta
                    lora_content_loss = lora_content_loss_sum / n_terms

                align_loss = None
                # TAL 计算并类加到梯度中
                if args.with_align_loss and base_capture["k"]:
                    align_sum, n_terms = 0.0, 0
                    mask_rare = rare_mask.unsqueeze(-1)
                    mask_class = super_mask.unsqueeze(-1)

                    for base_out, delta_out in zip(base_capture["k"], lora_content_capture['k']):
                        total_rare = (base_out + delta_out) * mask_rare
                        base_cls = base_out * mask_class

                        if mask_rare.any() and mask_class.any():
                            rare_vec = total_rare[mask_rare.squeeze(-1)].view(-1, total_rare.size(-1))
                            cls_vec = base_cls[mask_class.squeeze(-1)].view(-1, base_cls.size(-1))
                            align_sum += (rare_vec - cls_vec).abs().mean()
                            n_terms += 1

                    if n_terms > 0:
                        align_loss = align_sum / n_terms
                        loss = loss + args.align_loss_weight * align_loss
                
               
                gsa_loss = None
                if args.with_gsa_loss and len(lora_style_probs_capture) > 0:
                    gsa_sum, n_terms = 0.0, len(lora_style_probs_capture)
                    
                    for s in lora_style_probs_capture:
                        p = s / (s.sum(dim=-1, keepdim=True) + 1e-8)
                        spatial_len = p.size(-1)
                        uniform_target = torch.ones_like(p) / spatial_len

                        kl_div = F.kl_div(
                            p.log(),
                            uniform_target,
                            reduction="batchmean"
                        )
                        gsa_sum = gsa_sum + kl_div
                    
                    gsa_loss = gsa_sum / n_terms

                    if n_terms > 0:
                        loss = loss + args.gsa_loss_weight * gsa_loss
               
                for _lst in base_capture.values():
                    _lst.clear()
                for _lst in lora_content_capture.values():
                    _lst.clear()
                lora_style_probs_capture.clear()
                # -----------------------------------------------------------

                accelerator.backward(loss)

                """
                sampled_steps: 一个完整周期的步数
                sample_times: 总共做几轮 column separation
                gloabl_step: 当前训练步数
                """
                # 因为计算 cone 需要用到 grad 所以这里要在 backward 之后才能更新
                if args.with_period_column_separation: 
                    # * When finish one time sampling / Start next time sampling, we will do 
                    # * 1. Save the current mask 
                    # * 2. Let all of the mask as True ==> Since when start sampling we will use all of the columns
                    # * 3. Set all merger gradient as false 
                    # * 4. Set with_orthognal as False ==> When sampling doesn't optimize merger
                    
                    # 看完后面懂了，这里是对采样次数 * 每次修改 cone 的步数 后的步数如果还要的化就值微调选中的列。但是默认值又是 False
                    if global_step >= args.sample_times * sampled_steps:
                        # * a few step before finish: only train the overlap part 
                        lora_gradient_zeroout(unet, args.with_finetune_mask)
                    else:
                        # * start sampling and setting all merger matrix gradient to False
                        
                        # 用于判断当前处于 column seperation 的哪个周期
                        # 每个周期第一步
                        if (global_step) % sampled_steps == 0:
                            lora_merge_all_activate(unet, value=False) # merge 向量梯度关闭 
                            with_orthognal = False # 关闭损失
                            loss_orthognal = None
                        # * When accumulating sampling, add the cone value across steps
                        
                        #  cone 累积
                        elif (global_step % sampled_steps < num_update_steps_per_epoch): 
                            # 这里是采样阶段,是故意传入空字典的,只对 cone 进行累积
                            # 也就是这次执行只会来调用每层的 get_unziplora_cone 来累积 cone
                            if args.with_accumulate_cone: 
                                unet, _, _ = lora_merge_cone_select(unet, {}, {}, \
                                        logged=False, avoid=False, 
                                        accumulate=True)
                        # * When sample: we will do 
                        # * 1. compute cone based on current gradient, and store it as a score in class
                        # * 2. Set all of mask of the layers as false ==> Allowed column filter
                        # * 3. renew the filter based on score and previous filter
                        # * 4. set require_gradient for corresponding merger
                        # * 5. Set allowed cosine similarity between merger
                        
                        else: 
                            # * if just finish sampling => set mask and start train merger
                            if (global_step - num_update_steps_per_epoch) % sampled_steps == 0:
                                # 根据 cone累积做 topk 列选择，并且更新 mask
                                with torch.no_grad():
                                    # 这里就是真的更新了
                                    unet, content_cone_buffer, style_cone_buffer = lora_merge_cone_select(unet, mask_dictionary_style, mask_dictionary_content, \
                                        logged=args.with_grad_record, column_ratio=args.column_ratio, avoid=args.with_no_overlap_first, 
                                        accumulate=False)
                                    with_orthognal = True
                            else:
                                lora_gradient_zeroout(unet, args.with_finetune_mask)
                
                
                # 在 gradient_accumulation_steps > 1 时才有意义
                # 表示当前是梯度累积的最后一步，对所有 LoRA 参数做梯度裁剪，比秒梯度爆炸
                if accelerator.sync_gradients:
                    params_to_clip = (unet_params_to_optimze)
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    # 把 LoRA layer 中的 merge 向量固定在 [0, 1] 范围内
                    lora_merge_clamp(unet, "content")
                    lora_merge_clamp(unet, "style")
            # Checks if the accelerator has performed an optimization step behind the scenes


            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            
                            # 只保留罪行的 args.checkpoints_total_limit 个模型
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(output_dir=save_path)
                        logger.info(f"Saved state to {save_path}")
        
            logs = {
                "loss": loss.detach().item(),
                "reconstruction loss": reconstruction_loss.detach().item(),
                "lora_content_loss": lora_content_loss.detach().item() if lora_content_loss is not None else 0.0, 
                "align_loss": align_loss.detach().item() if align_loss is not None else 0.0,
                "gsa_loss": gsa_loss.detach().item() if gsa_loss is not None else 0.0,
                "lr": lr_scheduler.get_last_lr()[0]
            }

            # 记录数据
            with torch.no_grad():
                # quick_log 只是用来判断是否断开计算图
                # 为 True 是表示断开，单纯用于日志记录； 为 False 时表示不断开，可以用于反向传播，单实际上还是用于日志记录。
                style_norm_dict = lora_norm_log(unet, "style", quick_log=True)
                content_norm_dict = lora_norm_log(unet, "content", quick_log=True)
                content_merge_dict = lora_merge_log(unet, "content")
                style_merge_dict = lora_merge_log(unet, "style")
            
            # 字典的 update 会把 传入字典中的键值对加入到 logs 中，若键已经存在的话就会更新这个值
            logs.update(style_norm_dict)
            logs.update(content_norm_dict)
            logs.update(content_merge_dict)
            logs.update(style_merge_dict)
            
            if args.with_prior_preservation:
                if args.class_data_dir is not None:
                    logs["content prior loss"] = prior_loss_content.detach().item()
                if args.class_data_dir_2 is not None:
                    logs["style prior loss"] = prior_loss_style.detach().item()
            
            if with_orthognal and loss_orthognal is not None:
                logs["orthognal loss"] = loss_orthognal.detach().item()
            
            if args.with_grad_record and (global_step - 1 - num_update_steps_per_epoch) % sampled_steps == 0 and \
                args.with_period_column_separation and content_cone_buffer and style_cone_buffer:
                tracker.log(
                    {
                        "cone sparsity": [wandb.Image(content_cone_buffer), wandb.Image(style_cone_buffer)]
                    }
                )

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                if args.with_period_column_separation: 
                    # * We will save the filtered output
                    lora_merge_all_activate(unet, value=True)
                break

        # if accelerator.is_main_process:
        #     print("main process")
            if (
                args.validation_content is not None and args.validation_style is not None
                and args.heatmap_steps > 0 and (global_step - 1) % args.heatmap_steps == 0 and (global_step - 1) > 0
            ):
                if args.with_image_per_validation:
                    
                    # ----------------------------------------------------
                    # 清空 lora_content_capture 和 base_capture 的捕获的输出
                    ENABLE_LORA_CAPTURE = False
                    ENABLE_BASE_CAPTURE = False
                    ENABLE_STYLE_CAPTURE = False
                    for _lst in lora_content_capture.values():
                        _lst.clear()
                    for _lst in base_capture.values():
                        _lst.clear()
                    lora_style_probs_capture.clear()                    
                    # ----------------------------------------------------


                    # create pipeline
                    print("Create pipeline to upload validation")
                    if not args.train_text_encoder:
                        text_encoder_one = text_encoder_cls_one.from_pretrained(
                            args.pretrained_model_name_or_path,
                            subfolder="text_encoder",
                            revision=args.revision,
                        )
                        text_encoder_two = text_encoder_cls_two.from_pretrained(
                            args.pretrained_model_name_or_path,
                            subfolder="text_encoder_2",
                            revision=args.revision,
                        )
                        pipeline = StableDiffusionXLUnZipLoRAPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        vae=vae,
                        text_encoder=accelerator.unwrap_model(text_encoder_one),
                        text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                        unet=accelerator.unwrap_model(unet),
                        revision=args.revision,
                        torch_dtype=weight_dtype,
                    )

                    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
                    scheduler_args = {}

                    if "variance_type" in pipeline.scheduler.config:
                        variance_type = pipeline.scheduler.config.variance_type

                        if variance_type in ["learned", "learned_range"]:
                            variance_type = "fixed_small"

                        scheduler_args["variance_type"] = variance_type

                    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                        pipeline.scheduler.config, **scheduler_args
                    )

                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)
                    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
                    # run inference
                    logged_images = []
                    saved_images = []

                    images, _, heatmap_panels = log_validation(
                        pipeline, 
                        args, 
                        accelerator, 
                        args.validation_prompt, 
                        args.validation_prompt, 
                        args.validation_prompt,
                        generate_heatmap=False,
                        tokenizer_one=tokenizer_one,
                        tokenizer_two=tokenizer_two,
                        heatmap_type='content'
                    )
                    if len(heatmap_panels) > 0:
                        rare_heatmaps = [t[0] for t in heatmap_panels]
                        cls_heatmaps = [t[1] for t in heatmap_panels]
                    else:
                        rare_heatmaps, cls_heatmaps = [], []

                    if len(rare_heatmaps) > 0:
                        logged_images.append(concatenate_horizontal_img(rare_heatmaps))
                        logged_images.append(concatenate_horizontal_img(cls_heatmaps))
                    logged_images.append(concatenate_horizontal_img(images))
                    saved_images += images
                   
                    # 只生成 content 图片
                    if args.validation_prompt_content is not None:
                        unziplora_set_forward_type(unet, type="content")
                        images, _ , heatmap_panels = log_validation(pipeline, 
                            args, accelerator, 
                            args.validation_prompt_content, 
                            args.validation_prompt_content, 
                            args.validation_prompt_content,
                            generate_heatmap=False,
                            tokenizer_one=tokenizer_one,
                            tokenizer_two=tokenizer_two,
                            heatmap_type='content'
                        )
                        if len(heatmap_panels) > 0:
                            rare_heatmaps = [t[0] for t in heatmap_panels]
                            cls_heatmaps = [t[1] for t in heatmap_panels]
                        else:
                            rare_heatmaps, cls_heatmaps = [], []

                        if len(rare_heatmaps) > 0:
                            logged_images.append(concatenate_horizontal_img(rare_heatmaps))
                            logged_images.append(concatenate_horizontal_img(cls_heatmaps))
                        logged_images.append(concatenate_horizontal_img(images))
                        saved_images += images
                        unziplora_set_forward_type(unet, type="both")
                    
                    # 只生成 style 图片
                    if args.validation_prompt_style is not None:
                        unziplora_set_forward_type(unet, type="style")
                        images, _, _ = log_validation(pipeline, args, accelerator, 
                        args.validation_prompt_style, 
                        args.validation_prompt_style, 
                        validation_prompt_style=args.validation_prompt_style
                        )
                        logged_images.append(concatenate_horizontal_img(images))
                        saved_images += images
                        unziplora_set_forward_type(unet, type="both")

                    # 把 validation 过程中生成的图片上传 wandb
                    for tracker in accelerator.trackers:
                        phase_name = "validation"
                        print(f"Logged epoch {epoch}")
                        if tracker.name == "tensorboard":
                            tracker.writer.add_images(phase_name, logged_images, epoch, dataformats="NHWC")
                        if tracker.name == "wandb":
                            concatenated = concatenate_vertical_sublst(logged_images)
                            tracker.log(
                                {
                                    phase_name: wandb.Image(concatenated)
                                }
                            )

                    # ----------------------------------------------------
                    # 验证完之后就重新开启捕获 content LoRA 和 base 的输出
                    ENABLE_LORA_CAPTURE = True
                    ENABLE_BASE_CAPTURE = True
                    ENABLE_STYLE_CAPTURE = True
                    for _lst in lora_content_capture.values():
                        _lst.clear()
                    for _lst in base_capture.values():
                        _lst.clear()
                    lora_style_probs_capture.clear()
                    # ----------------------------------------------------
                
                if args.with_saved_per_validation:
                    unet = accelerator.unwrap_model(unet)
                    unet = unet.to(torch.float32)
                    unet_lora_layers_content, unet_lora_layers_merger_content, unet_lora_layers_base_weight_content  = unet_inverse_ziplora_state_dict(unet, key="content")
                    unet_lora_layers_style, unet_lora_layers_merger_style, unet_lora_layers_base_weight_style = unet_inverse_ziplora_state_dict(unet, key="style")

                    if args.train_text_encoder:
                        text_encoder_one = accelerator.unwrap_model(text_encoder_one)
                        text_encoder_lora_layers = text_encoder_lora_state_dict(
                            text_encoder_one.to(torch.float32)
                        )
                        text_encoder_two = accelerator.unwrap_model(text_encoder_two)
                        text_encoder_2_lora_layers = text_encoder_lora_state_dict(
                            text_encoder_two.to(torch.float32)
                        )
                    else:
                        text_encoder_lora_layers = None
                        text_encoder_2_lora_layers = None
                    
                    torch.save(unet_lora_layers_merger_content, f"{args.output_dir}_{global_step}_merger_content.pth")
                    torch.save(unet_lora_layers_base_weight_content, f"{args.output_dir}_{global_step}_base_weight_content.pth")
                    StableDiffusionXLPipeline.save_lora_weights(
                        save_directory=f"{args.output_dir}_{global_step}_content",
                        unet_lora_layers=unet_lora_layers_content,
                        text_encoder_lora_layers=text_encoder_lora_layers,
                        text_encoder_2_lora_layers=text_encoder_2_lora_layers,
                    )
                    torch.save(unet_lora_layers_merger_style, f"{args.output_dir}_{global_step}_merger_style.pth")
                    torch.save(unet_lora_layers_base_weight_style, f"{args.output_dir}_{global_step}_base_weight_style.pth")
                    StableDiffusionXLPipeline.save_lora_weights(
                        save_directory=f"{args.output_dir}_{global_step}_style",
                        unet_lora_layers=unet_lora_layers_style,
                        text_encoder_lora_layers=text_encoder_lora_layers,
                        text_encoder_2_lora_layers=text_encoder_2_lora_layers,
                    )

                    # unet.to(accelerator.device)
                    del unet_lora_layers_content
                    del unet_lora_layers_style
                    del unet_lora_layers_merger_content
                    del unet_lora_layers_merger_style
                    del text_encoder_lora_layers
                    del text_encoder_2_lora_layers
                    # torch.cuda.empty_cache()
    

    # Exit Training
    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # 打包模型
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_layers_content, unet_lora_layers_merger_content, unet_lora_layers_base_weight_content = unet_inverse_ziplora_state_dict(unet, key="content")
        unet_lora_layers_style, unet_lora_layers_merger_style, unet_lora_layers_base_weight_style = unet_inverse_ziplora_state_dict(unet, key="style")

        if args.train_text_encoder:
            text_encoder_one = accelerator.unwrap_model(text_encoder_one)
            text_encoder_lora_layers = text_encoder_lora_state_dict(
                text_encoder_one.to(torch.float32)
            )
            text_encoder_two = accelerator.unwrap_model(text_encoder_two)
            text_encoder_2_lora_layers = text_encoder_lora_state_dict(
                text_encoder_two.to(torch.float32)
            )
        else:
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None

        # 保存 content_lora_layer
        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=f"{args.output_dir}_content",
            unet_lora_layers=unet_lora_layers_content,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )
        # 保存 content_merge 向量
        torch.save(unet_lora_layers_merger_content, f"{args.output_dir}_merger_content.pth")
        # 保存 base weight content 矩阵
        torch.save(unet_lora_layers_base_weight_content, f"{args.output_dir}_base_weight_content.pth")
        
        # 保存 style_lora_layer
        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=f"{args.output_dir}_style",
            unet_lora_layers=unet_lora_layers_style,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )
        # 保存 style_merge 向量
        torch.save(unet_lora_layers_merger_style, f"{args.output_dir}_merger_style.pth")
        # 保存 base weight style 矩阵
        torch.save(unet_lora_layers_base_weight_style, f"{args.output_dir}_base_weight_style.pth")
        
        # remove unuse models for save GPU memory
        unet = unet.cpu()
        text_encoder_one = text_encoder_one.cpu()
        text_encoder_two = text_encoder_two.cpu()
        del text_encoder_one, text_encoder_two
        del optimizer
        if args.train_text_encoder:
            del text_encoder_lora_layers, text_encoder_2_lora_layers

        # Final inference
        # Load previous pipeline
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
            revision=args.revision,
            torch_dtype=weight_dtype,
        )
        pipeline = load_pipeline_from_sdxl(
            args.pretrained_model_name_or_path,
            vae=vae,
        )

        # load attention processors
        pipeline.unet = insert_unziplora_to_unet(pipeline.unet, f"{args.output_dir}_content", f"{args.output_dir}_style", \
                weight_content_path=f"{args.output_dir}_merger_content.pth" , weight_style_path=f"{args.output_dir}_merger_style.pth", \
                rank=args.rank, device=accelerator.device)
        # pipeline.unet = unet
        del unet

        # run inference
        logged_images = []
        saved_images = []
        if args.validation_content and args.validation_style and args.num_validation_images > 0:
            pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)
            
            # 生成推理图片
            images, image_list, heatmap_panels = log_validation(
                pipeline,
                args,
                accelerator,
                args.validation_prompt,
                args.validation_prompt,
                args.validation_prompt,
                generate_heatmap=False,
                tokenizer_one=tokenizer_one,
                tokenizer_two=tokenizer_two,
                heatmap_type='content'
            )
            if len(heatmap_panels) > 0:
                rare_heatmaps = [t[0] for t in heatmap_panels]
                cls_heatmaps = [t[1] for t in heatmap_panels]
            else:
                rare_heatmaps, cls_heatmaps = [], []
            
            if len(rare_heatmaps) > 0:
                logged_images.append(concatenate_horizontal_img(rare_heatmaps))
                logged_images.append(concatenate_horizontal_img(cls_heatmaps))
            # 把单张图片拼成一行
            logged_images.append(concatenate_horizontal_img(images))
            saved_images += images

        # 只在有 content_prompt 下生成图片
        if args.validation_prompt_content and args.num_validation_images > 0:
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=vae,
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
            pipeline.load_lora_weights(f"{args.output_dir}_content")
            pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)
            
            images, image_list, heatmap_panels = log_validation(
                pipeline,
                args,
                accelerator,
                args.validation_prompt_content,
                generate_heatmap=False,
                tokenizer_one=tokenizer_one,
                tokenizer_two=tokenizer_two,
                heatmap_type='content'
            )
            
            if len(heatmap_panels) > 0:
                rare_heatmaps = [t[0] for t in heatmap_panels]
                cls_heatmaps = [t[1] for t in heatmap_panels]
            else:
                rare_heatmaps, cls_heatmaps = [], []

            if len(rare_heatmaps) > 0:
                logged_images.append(concatenate_horizontal_img(rare_heatmaps))
                logged_images.append(concatenate_horizontal_img(cls_heatmaps))
            logged_images.append(concatenate_horizontal_img(images))
            saved_images += images
        
        # 只在有 style_prompt 下生成图片
        if args.validation_prompt_style and args.num_validation_images > 0:
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=vae,
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
            pipeline.load_lora_weights(f"{args.output_dir}_style")
            pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)
            
            images, image_list, heatmap_panels = log_validation(
                pipeline,
                args,
                accelerator,
                args.validation_prompt_style,
                generate_heatmap=False,
                tokenizer_one=tokenizer_one,
                tokenizer_two=tokenizer_two,
                heatmap_type='style'
            )
            
            if len(heatmap_panels) > 0:
                rare_heatmaps = [t[0] for t in heatmap_panels]
                cls_heatmaps = [t[1] for t in heatmap_panels]
            else:
                rare_heatmaps, cls_heatmaps = [], []

            if len(rare_heatmaps) > 0:
                logged_images.append(concatenate_horizontal_img(rare_heatmaps))
                logged_images.append(concatenate_horizontal_img(cls_heatmaps))
            
            logged_images.append(concatenate_horizontal_img(images))
            saved_images += images
        
        for tracker in accelerator.trackers:
            phase_name = "test"
            if tracker.name == "tensorboard":
                tracker.writer.add_images(phase_name, logged_images, epoch, dataformats="NHWC")
            if tracker.name == "wandb":
                concatenated = concatenate_vertical_sublst(logged_images)
                tracker.log(
                    {
                        phase_name: wandb.Image(concatenated)
                    }
                )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=saved_images,
                base_model=args.pretrained_model_name_or_path,
                train_text_encoder=args.train_text_encoder,
                instance_prompt=instance_prompt,
                validation_prompt=f"{args.validation_content} {args.validation_style}",
                repo_folder=args.output_dir,
                vae_path=args.pretrained_vae_model_name_or_path,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
