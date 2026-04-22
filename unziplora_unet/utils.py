import os
from typing import Optional, Dict, Union
from huggingface_hub import hf_hub_download
import copy 
import math

import torch
import torch.nn.functional as F

from safetensors import safe_open
from diffusers.loaders.lora import LORA_WEIGHT_NAME_SAFE
import itertools

from unziplora_unet.unziplora_linear_layer import UnZipLoRALinearLayer, UnZipLoRALinearLayerInfer
from unziplora_unet.pipeline_stable_diffusion_xl import StableDiffusionXLUnZipLoRAPipeline
from unziplora_unet.unet_2d_condition import UNet2DConditionModel

from diffusers import (
     DDPMScheduler, 
     AutoencoderKL,
    )
from transformers import (
    AutoTokenizer, 
    CLIPTextModel, 
    CLIPTextModelWithProjection,
    )
from record_utils.cone import cone_matrix, cone_column_sparsity, draw_concatenated_heatmap

# 绘制热力图函数
from matplotlib import colormaps
from PIL import Image
import numpy as np

universal_nevigate = [
"watermark", "lowres", "low quality",
"blur", "out of focus", "grainy", "jpeg artifacts",
"cropped", "poorly lit", "duplicate",
]

SDXL_group_list = [
    "up_blocks.0.attentions.0.",
    "up_blocks.0.attentions.1.",
    "up_blocks.0.attentions.2.",
    "up_blocks.1.attentions.0.",
    "up_blocks.1.attentions.1.",
    "up_blocks.1.attentions.2.",
    "down_blocks.1.attentions.0.",
    "down_blocks.1.attentions.1.",
    "down_blocks.2.attentions.0.",
    "down_blocks.2.attentions.1.",
    "mid_block.attentions.0.",
]

SDXL_up_layer_dictionary = {
    1: "up_blocks.0.attentions.1.",
    2: "up_blocks.0.attentions.1.",
    3: "up_blocks.0.attentions.1.",
    4: "up_blocks.0.attentions.1.",
    5: "up_blocks.0.attentions.1.",
    6: "up_blocks.0.attentions.1.",
    7: "up_blocks.0.attentions.1.",
    8: "up_blocks.0.attentions.1.",
    9: "up_blocks.0.attentions.1.",
    10: "up_blocks.0.attentions.1.",
    11: "up_blocks.1.attentions.0.",
    12: "up_blocks.1.attentions.0.",
    13: "up_blocks.1.attentions.1.",
    14: "up_blocks.1.attentions.1.",
    15: "up_blocks.1.attentions.2.",
    16: "up_blocks.1.attentions.2.",
    17: "up_blocks.0.attentions.2.",
    18: "up_blocks.0.attentions.2.",
    19: "up_blocks.0.attentions.2.",
    20: "up_blocks.0.attentions.2.",
    21: "up_blocks.0.attentions.2.",
    22: "up_blocks.0.attentions.2.",
    23: "up_blocks.0.attentions.2.",
    24: "up_blocks.0.attentions.2.",
    25: "up_blocks.0.attentions.2.",
    26: "up_blocks.0.attentions.2.",
    27: "up_blocks.0.attentions.0.",
    28: "up_blocks.0.attentions.0.",
    29: "up_blocks.0.attentions.0.",
    30: "up_blocks.0.attentions.0.",
    31: "up_blocks.0.attentions.0.",
    32: "up_blocks.0.attentions.0.",
    33: "up_blocks.0.attentions.0.",
    34: "up_blocks.0.attentions.0.",
    35: "up_blocks.0.attentions.0.",
    36: "up_blocks.0.attentions.0.",
}

logged_layers = [
        "down_blocks.1.attentions.0.transformer_blocks.0",
        "down_blocks.1.attentions.1.transformer_blocks.0",
        "down_blocks.2.attentions.0.transformer_blocks.5",
        "down_blocks.2.attentions.0.transformer_blocks.9",
        "down_blocks.2.attentions.1.transformer_blocks.2",
        "down_blocks.2.attentions.1.transformer_blocks.3",
        "up_blocks.0.attentions.0.transformer_blocks.8",
        "up_blocks.0.attentions.0.transformer_blocks.3",
        "up_blocks.0.attentions.1.transformer_blocks.1",
        "up_blocks.0.attentions.1.transformer_blocks.9",
        "up_blocks.0.attentions.2.transformer_blocks.0",
        "up_blocks.0.attentions.2.transformer_blocks.4",
        "up_blocks.1.attentions.0.transformer_blocks.0",
        "up_blocks.1.attentions.1.transformer_blocks.0",
        "up_blocks.1.attentions.2.transformer_blocks.0",   
    ]

SDXL_content_layer_mask = {
    "up_blocks.0.attentions.0.": False,
    "up_blocks.0.attentions.1.": False,
    "up_blocks.0.attentions.2.": False,
    "up_blocks.1.attentions.0.": False,
    "up_blocks.1.attentions.1.": False,
    "up_blocks.1.attentions.2.": False,
    "down_blocks.2.attentions.0.": False,
    "down_blocks.2.attentions.1.": False,
    "down_blocks.1.attentions.1.": False,
    "down_blocks.1.attentions.0.": False,
    "mid_block.attentions.0.": False,
}

SDXL_style_layer_mask = {
    "up_blocks.0.attentions.0.": False,
    "up_blocks.0.attentions.1.": False,
    "up_blocks.0.attentions.2.": False,
    "up_blocks.1.attentions.0.": False,
    "up_blocks.1.attentions.1.": False,
    "up_blocks.1.attentions.2.": False,
}

global_seeds = [0, 111, 1111, 1000, 1234, 123, 101, 42, 24, 26, 53]

def get_lora_weights(
    lora_name_or_path: str, subfolder: Optional[str] = None, **kwargs
) -> Dict[str, torch.Tensor]:
    """
    get lora weights from saved path 
    
    Args:
        lora_name_or_path (str): huggingface repo id or folder path of lora weights
        subfolder (Optional[str], optional): sub folder. Defaults to None.
    """
    if os.path.exists(lora_name_or_path):
        if subfolder is not None:
            lora_name_or_path = os.path.join(lora_name_or_path, subfolder)
        if os.path.isdir(lora_name_or_path):
            lora_name_or_path = os.path.join(lora_name_or_path, LORA_WEIGHT_NAME_SAFE)
    else:
        lora_name_or_path = hf_hub_download(
            repo_id=lora_name_or_path,
            filename=LORA_WEIGHT_NAME_SAFE,
            subfolder=subfolder,
            **kwargs,
        )
    assert lora_name_or_path.endswith(
        ".safetensors"
    ), "Currently only safetensors is supported"
    tensors = {}
    with safe_open(lora_name_or_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def unziplora_set_forward_type(unet: UNet2DConditionModel, type: str = "both"):
    """
    set forward type for the network: using either content part style part or both
    """
    assert type in ["both", "content", "style"]

    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                assert hasattr(lora_layer, "set_forward"), lora_layer
                lora_layer.set_forward(type)
    return unet

# * Functions about insert blocked mask
def generate_mask_in_unet(mask_dictionary={}):
    """
    based on the masked dictionary, generate the blocked parts for UNet
    """
    masked_layers = {}
    for key, value in mask_dictionary.items():
        block_name = key 
        for pattern in value:
            element = pattern.split("_")
            # 那 mask_dictionary 中的第一个 key：value 来模拟
            # element = ['N', '0', 'A', 'A']
            if element[0] == "N":
                block_nums = ['']
            elif element[0] == "A":
                if key == "up_blocks.":
                    block_nums = ['0', '1']
                elif key == "down_blocks.":
                    block_nums = ['1', '2']
            else:
                block_nums = element[0].split(',')
                
            
            if element[1] == "A":
                if key == "up_blocks.":
                    group_nums = ['0', '1', '2']
                elif key == "down_blocks.":
                    group_nums = ['1', '0']
            else:
                group_nums = element[1].split(',')
            
            if element[2] == "A":
                attention_types = ["attn1", "attn2"]
            else:
                attention_types = [f"attn{i}" for i in element[2].split(',')]
                
            if element[3] == "A":
                module_types = ["to_k", "to_v", "to_q", "to_out.0"]
            else:
                module_types = [f"to_{i}" for i in element[3].split(',')]

            # 计算笛卡尔积
            key_combinations = itertools.product(
                block_nums, group_nums
            )
            value_combinations = [f"{attention_type}.{module_type}" \
                for attention_type, module_type in itertools.product(
                attention_types, module_types)]
            
            for block_num, group_num in key_combinations:
                masked_layers_key = f"{block_name}{block_num}.attentions.{group_num}."
                if masked_layers_key not in masked_layers.keys():
                    masked_layers[masked_layers_key] = copy.deepcopy(value_combinations)
                else:
                    masked_layers[masked_layers_key] += value_combinations
    return masked_layers

def insert_layer_dic_mask(unet, masked_layers, key, set_value=False, insert_toggle=False, toggle_key="content"):
    """
    Given the masked layers generated by "generate_mask_in_unet" 
    set the hard mask for the blocks for block separation
    """
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Module) and 'attn' in name:
                                                                                 
                before_key = name.split("transformer")[0]
                if before_key in masked_layers.keys():
                    attention_name = "attn" + name.split("attn")[-1]
                    if attention_name in masked_layers[before_key]:
                        lora_layer = getattr(module, "lora_layer")
                        if lora_layer is not None:
                            assert hasattr(lora_layer, "set_layer_mask"), lora_layer
                            lora_layer.set_layer_mask(key)
                            if set_value:
                                lora_layer.set_parameter_zero(key)
                    elif insert_toggle:
                        lora_layer = getattr(module, "lora_layer")
                        if lora_layer is not None:
                            assert hasattr(lora_layer, "set_layer_mask"), lora_layer
                            lora_layer.set_layer_mask(toggle_key)
                            if set_value:
                                lora_layer.set_parameter_zero(toggle_key)
                elif insert_toggle:
                    lora_layer = getattr(module, "lora_layer")
                    if lora_layer is not None:
                        assert hasattr(lora_layer, "set_layer_mask"), lora_layer
                        lora_layer.set_layer_mask(toggle_key)
                        if set_value:
                            lora_layer.set_parameter_zero(toggle_key)
    return unet
def insert_mask(unet: UNet2DConditionModel, key, mask_dictionary=None, set_value=False, insert_toggle=False, toggle_key="content"):
    """
    set the hard mask for the blocks for block separation
    """
    masked_layers = generate_mask_in_unet(mask_dictionary)
    unet = insert_layer_dic_mask(unet, masked_layers, key, set_value, insert_toggle, toggle_key)
    return unet


def inverse_ziplora_compute_weight_similarity(unet):
    '''
    Compute the cosine similarity between two merge matrix
    '''
    similarities = []
    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                assert hasattr(lora_layer, "compute_mergers_similarity"), lora_layer
                attn_module = name.split(".")[-1]
                # if attn_module == "to_v" or attn_module == "to_k":
                similarities.append(lora_layer.compute_mergers_similarity())
                # print(similarities)
    # similarity = torch.stack(similarities).sum(dim=0)
    similarity = torch.stack(similarities).mean(dim=0)
    return similarity.squeeze()

def unet_inverse_ziplora_state_dict(
    unet: UNet2DConditionModel, key: str, quick_release: bool = False
) -> Dict[str, torch.Tensor]:
    r"""
    Returns:
        A state dict containing the state with the input key
    Usage:
        For save the state dict of input
    """
    lora_state_dict = {}
    mask_state_dict = {}
    base_state_dict = {}
    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                assert hasattr(lora_layer, "get_unziplora_weight"), lora_layer
                weight_down, weight_up = lora_layer.get_unziplora_weight(key)
                lora_state_dict[f"unet.{name}.lora.up.weight"] = weight_up.contiguous()
                lora_state_dict[f"unet.{name}.lora.down.weight"] = weight_down.contiguous()
                merge_matrix = lora_layer.get_merger_mask(key)
                mask_state_dict[f"unet.{name}.lora.merge_{key}"] = merge_matrix.contiguous()
                
                base_weight_down, base_weight_up = lora_layer.get_base_lora_weight(key)
                base_state_dict[f'unet.{name}.lora.up.base_{key}'] = base_weight_up.contiguous()
                base_state_dict[f'unet.{name}.lora.down.base_{key}'] = base_weight_down.contiguous()
                if quick_release:
                    lora_layer.cpu()
    return lora_state_dict, mask_state_dict, base_state_dict


def initialize_unziplora_layer_for_inference(state_dict_content_down, state_dict_content_up, \
    state_dict_style_down, state_dict_style_up, \
    state_dict_merger_content, state_dict_merger_style, \
    part, **model_kwargs):
    '''
    Insert unziplora to inference
    '''
    unziplora_layer = UnZipLoRALinearLayerInfer(**model_kwargs)
    if state_dict_merger_content is not None and state_dict_merger_style is not None:
        unziplora_layer.load_state_dict(
            {
                "lora_matrix_dic.content_down.weight": state_dict_content_down[part],
                "lora_matrix_dic.content_up.weight": state_dict_content_up[part],
                "lora_matrix_dic.style_down.weight": state_dict_style_down[part],
                "lora_matrix_dic.style_up.weight": state_dict_style_up[part],
                "merge_content": state_dict_merger_content[part],
                "merge_style": state_dict_merger_style[part]
            },
            strict=False,
        )
    else:
        unziplora_layer.load_state_dict(
            {
                "lora_matrix_dic.content_down.weight": state_dict_content_down[part],
                "lora_matrix_dic.content_up.weight": state_dict_content_up[part],
                "lora_matrix_dic.style_down.weight": state_dict_style_down[part],
                "lora_matrix_dic.style_up.weight": state_dict_style_up[part],
            },
            strict=False,
        )
    return unziplora_layer

def use_lora_weights_for_inference(
    tensors: Dict[str, torch.Tensor], key: str, prefix: str = "unet.unet."
) -> Dict[str, torch.Tensor]:
    """
    Initialize the saved model value for down and up LoRA matrix
    Args:
        tensors (torch.Tensor): state dict of lora weights
        key (str): target attn layer's key
        prefix (str, optional): prefix for state dict.Defaults to "unet.unet.".
    """
    target_key = prefix + key
    out_down = {}
    out_up = {}
    # print(tensors.keys())
    for part in ["to_q", "to_k", "to_v", "to_out.0"]:
        down_key = target_key + f".{part}.lora.down.weight"
        up_key = target_key + f".{part}.lora.up.weight"
        out_down[part] = tensors[down_key]
        out_up[part] = tensors[up_key]
    return out_down, out_up

def use_lora_mergers_for_inference(
    tensors_content: Dict[str, torch.Tensor], tensors_style: Dict[str, torch.Tensor], key: str, prefix: str = "unet.unet."
) -> Dict[str, torch.Tensor]:
    """
    Initialize the saved model value for content and style soft mask merger
    Args:
        tensors (torch.Tensor): state dict of lora weights
        key (str): target attn layer's key
        prefix (str, optional): prefix for state dict.Defaults to "unet.unet.".
    """
    target_key = prefix + key
    merger_content  = {}
    merger_style  = {}
    # print(tensors.keys())
    for part in ["to_q", "to_k", "to_v", "to_out.0"]:
        merge_content = target_key + f".{part}.lora.merge_content"
        merge_style = target_key + f".{part}.lora.merge_style"
        merger_content[part] = tensors_content[merge_content]
        merger_style[part] = tensors_style[merge_style]
    return merger_content, merger_style

def insert_unziplora_to_unet(
    unet: UNet2DConditionModel, content_lora_path: str, style_lora_path: str, weight_content_path: str = None, weight_style_path: str = None, \
        rank: int = 64, device: Optional[Union[torch.device, str]] = None, **kwargs
):
    """
    Initialize the saved model value for content and style soft mask merger
    Args:
        tensors (torch.Tensor): state dict of lora weights
        key (str): target attn layer's key
        prefix (str, optional): prefix for state dict.Defaults to "unet.unet.".
    """
    tensors_content = get_lora_weights(content_lora_path, **kwargs)
    tensors_style = get_lora_weights(style_lora_path, **kwargs)
    
    # 这里是加载merge 向量
    if weight_content_path is not None:
        weight_content_state_dict = torch.load(weight_content_path)
    if weight_style_path is not None:
        weight_style_state_dict = torch.load(weight_style_path)
    # weight_state_dict = {key: value.to(device) for key, value in weight_state_dict.items()}
    
    # 吧
    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # Parse the attention module.
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)
        # Get prepared for ziplora
        attn_name = ".".join(attn_processor_name.split(".")[:-1])

        # 根据名字来拼接

        # 返回每个 attention layer 对应 lora 的权重
        state_dict_content_down, state_dict_content_up = use_lora_weights_for_inference(tensors_content, key=attn_name)
        state_dict_style_down, state_dict_style_up = use_lora_weights_for_inference(tensors_style, key=attn_name)
        
        if weight_content_path is not None and weight_style_path is not None:
            # 返回融合之后的权重
            state_dict_merge_content, state_dict_merge_style = use_lora_mergers_for_inference(weight_content_state_dict, weight_style_state_dict, key=attn_name, prefix="unet.")
        else:
            state_dict_merge_style = None
            state_dict_merge_content = None
        # Set the `lora_layer` attribute of the attention-related matrices.
        
        # 将一个attention layer 中的所有 lora lora 参数都加载到一个字典中
        kwargs = {
            "state_dict_content_down": state_dict_content_down, 
            "state_dict_content_up": state_dict_content_up,
            "state_dict_style_down": state_dict_style_down,
            "state_dict_style_up": state_dict_style_up,
            "state_dict_merger_content": state_dict_merge_content,
            "state_dict_merger_style": state_dict_merge_style 
        }
        
        attn_module.to_q.set_lora_layer(
            # 加载 UnZipLoRAInferLayer 和 训练的过程中保存的参数基本一致
            initialize_unziplora_layer_for_inference(
                in_features=attn_module.to_q.in_features,
                out_features=attn_module.to_q.out_features,
                part="to_q",
                lora_matrix_key=["content", "style"],
                rank=rank,
                device=device,
                **kwargs,
            )
        )
        attn_module.to_k.set_lora_layer(
            initialize_unziplora_layer_for_inference(
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
                part="to_k",
                lora_matrix_key=["content", "style"],
                device=device,
                rank=rank,
                **kwargs,
            )
        )
        attn_module.to_v.set_lora_layer(
            initialize_unziplora_layer_for_inference(
                in_features=attn_module.to_v.in_features,
                out_features=attn_module.to_v.out_features,
                part="to_v",
                device=device,
                lora_matrix_key=["content", "style"],
                rank=rank,
                **kwargs,
            )
        )
        attn_module.to_out[0].set_lora_layer(
            initialize_unziplora_layer_for_inference(
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                part="to_out.0",
                device=device,
                lora_matrix_key=["content", "style"],
                rank=rank,
                **kwargs,
            )
        )
    return unet

def load_pipeline_from_sdxl(MODEL_ID,
    tokenizer_one: AutoTokenizer=None,
    tokenizer_two: AutoTokenizer=None,
    vae: AutoencoderKL=None,
    unet: UNet2DConditionModel=None,
    noise_scheduler: DDPMScheduler=None,
    text_encoder_one: CLIPTextModel=None, 
    text_encoder_two: CLIPTextModelWithProjection=None, 
    ):
    """
    log UnZipLoRA sdxl pipeline, enabling use
    """
    if tokenizer_one is None:
        tokenizer_one = AutoTokenizer.from_pretrained(
        MODEL_ID,
        subfolder="tokenizer",
        revision=None
        )
    if tokenizer_two is None:
        tokenizer_two = AutoTokenizer.from_pretrained(
            MODEL_ID,
            subfolder="tokenizer_2",
            revision=None
        )
    if vae is None:
        vae = AutoencoderKL.from_pretrained(
                MODEL_ID,
                subfolder="vae",
                revision=None
            )
    if unet is None:
        unet = UNet2DConditionModel.from_pretrained(
            MODEL_ID, subfolder="unet", revision=None
        )
    if noise_scheduler is None:
        noise_scheduler = DDPMScheduler.from_pretrained(
            MODEL_ID, subfolder="scheduler"
        )
    if text_encoder_one is None:
        text_encoder_one = CLIPTextModel.from_pretrained(
            MODEL_ID,
            subfolder="text_encoder",
            revision=None,
        )
    if text_encoder_two is None:
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            MODEL_ID,
            subfolder="text_encoder_2",
            revision=None,
        )
    pipeline = StableDiffusionXLUnZipLoRAPipeline(
        vae = vae,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        unet=unet,
        scheduler=noise_scheduler,
    )
    return pipeline

# * Functions that help record statistics / record in wandb

'''
Get the current value of matrix
'''
def lora_original_value(unet: UNet2DConditionModel, key):
    original_mat = {}
    with torch.no_grad():
        for name, module in unet.named_modules():
            if hasattr(module, "set_lora_layer"):
                lora_layer = getattr(module, "lora_layer")
                if lora_layer is not None:
                    assert hasattr(lora_layer, "get_unziplora_weight"), lora_layer
                    weight_down, weight_up = lora_layer.get_unziplora_weight(key)
                    original_mat[name] = copy.deepcopy(weight_down.T @ weight_up.T)
    return original_mat

'''
clamp merger
'''
def lora_merge_clamp(unet: UNet2DConditionModel, key):
    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                assert hasattr(lora_layer, "clamp_merger"), lora_layer
                lora_layer.clamp_merger(key)

'''
Log norm value block wise
'''

# 就是给同一个 CrossAttentionBlock 下的 多个transfomer_block 的多个 attn 子模块都会被归到同一 before_key 下
def lora_norm_log(unet: UNet2DConditionModel, key, quick_log=False, dim="L2", multiple=False):
    norm_dict = {}
    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            before_key = f"{key}_{name.split('transformer')[0]}_norm"
            if lora_layer is not None:
                assert hasattr(lora_layer, "get_unziplora_norm"), lora_layer
                if before_key in norm_dict.keys():
                    norm_dict[before_key].append(lora_layer.get_unziplora_norm(key, dim=dim, quick_log=quick_log, multiple=multiple))
                else:
                    norm_dict[before_key] = [lora_layer.get_unziplora_norm(key, dim=dim, quick_log=quick_log, multiple=multiple)]
    norm_dict = {key: torch.mean(torch.tensor(value)) for key,value in norm_dict.items()}
    return norm_dict

'''
Log merge value block wise
'''
def lora_merge_log(unet: UNet2DConditionModel, key, quick_log=False):
    norm_dict = {}
    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            before_key = f"{key}_{name.split('transformer')[0]}_merge"
            if lora_layer is not None:
                assert hasattr(lora_layer, f"merge_{key}"), lora_layer
                if before_key in norm_dict.keys():
                    norm_dict[before_key] += torch.mean(getattr(lora_layer, f"merge_{key}"))
                else:
                    norm_dict[before_key] = torch.mean(getattr(lora_layer, f"merge_{key}"))
    return norm_dict

def lora_merge_all_activate(unet: UNet2DConditionModel, value=True):
# * If value => True: will use filter       + not train merger gradient 
# * if value => False:will not use filter   + not train merger gradient
    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                assert hasattr(lora_layer, "set_layer_mask"), lora_layer
                lora_layer.set_layer_mask("content", value)
                lora_layer.set_layer_mask("style", value)
                lora_layer.set_merger_gradient("content")
                lora_layer.set_merger_gradient("style")

def lora_cone_spectrum_log(merge_gradient, merge_weight):
    '''
    input: current value, current gradient
    output: logged image
    '''
    logged_layers_key = [f"{layers}.attn2.to_k" for layers in logged_layers]
    logged_gradient = {key: value  for key, value in merge_gradient.items() if key in logged_layers_key}
    logged_weight = {key: value  for key, value in merge_weight.items() if key in logged_layers_key}
    cone = cone_matrix(logged_gradient, logged_weight)
    cone_heatmap_matrix = cone_column_sparsity(cone)
    buffer = draw_concatenated_heatmap(cone_heatmap_matrix)
    return buffer

def lora_merge_cone_select(unet: UNet2DConditionModel, mask_dictionary_style={}, mask_dictionary_content={}, logged=False, \
    column_ratio=0.05, avoid=True, accumulate=True):
    '''
    Initialize the saved model value for content and style soft mask merger
    Args:
        mask_dictionary_style/content: hard mask for block separation
        logged (optional): Whether saved the cone in wandb
        column_ratio: the ratio of selected/trained column features in each step 
        avoid: whether avoid training the same features
        accumulate: whether to add the computed cone in the attributes of each layer
    '''
    
    # 生成 block seperation 的 mask 映射， 将 block seperation 的配置字典解析成具体的 attention 层名
    masked_layers_style = generate_mask_in_unet(mask_dictionary_style)
    masked_layers_content = generate_mask_in_unet(mask_dictionary_content)
    
    # the layers that will be logged to wandb
    logged_layers_key = [f"{layers}.attn2.to_k" for layers in logged_layers]
    
    # * blocked layers are blocks that are not inserted any all blocks filters
    # * masked_layers_style: blocks that are only with columns; other layers will insert all block filters
    # * block inserted with "all block" will multiply with masked .. / all columns will be used
    
    # 取 content 和 style 两个 mask 的交集，也就是即 content 和 style 都声明了的层。
    # 这就表示该层同时被两边的 block 分离约束， 所以不做 column seperation
    blocked_layers_key = masked_layers_content.keys() & masked_layers_style.keys()
    
    # 删除掉交集层,意思是这类层不交给任意一边，而是使用列分离
    blocked_layers = {key: masked_layers_content[key] for key in blocked_layers_key}
    for key in blocked_layers_key:
        del masked_layers_content[key]
        del masked_layers_style[key]
    
    logged_cone_layer_content = {}
    logged_cone_layer_style = {}
    
    # 遍历 unet 所有的 attention block
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Module) and 'attn' in name:
            if hasattr(module, "set_lora_layer"):                                                                
                lora_layer = getattr(module, "lora_layer")
                if lora_layer is not None:
                    # name 通常都是，name = "mid_block.attentions.0.transformer_blocks.0.attn2.to_k"
                    #  提取 block 前缀
                    # before_key 也就是 "up_blocks.0.attentions.2."， 然后就用 before key 去查找 masked_layers_style，masked_layers_content， blocked_layers
                    before_key = name.split("transformer")[0]
                    masked_key = None 
                    
                    # 如果 block 前缀在 content lora 和 style lora 的交集层中
                    # 且 attention 匹配，两百年都选
                    if before_key in blocked_layers.keys():
                        # 首先查看当前所在的 Attention Block 是否有被加入没有的话，自然也就不用判断了
                        
                        # 若加入了， 则判断当前 Attention Layer 是否有加入 blocked_layers 也就是是否同时被content 和 style 选中
                        attention_name = "attn" + name.split("attn")[-1]
                        if attention_name in blocked_layers[before_key]:
                            masked_key = None
                    
                    elif before_key in masked_layers_style.keys():
                        # 如果是只在 masked_layers_style 中
                       
                        attention_name = "attn" + name.split("attn")[-1]
                        
                        # 当前 attention layer 在 masked_layers_style 就将 masked_key 赋值为 style
                        # 也就是说当前的 layer 是被 style 独占的
                        if attention_name in masked_layers_style[before_key]:
                            masked_key = "style"
                    elif before_key in masked_layers_content.keys():
                        #这个判断和上面的判断同理 

                        attention_name = "attn" + name.split("attn")[-1]
                        if attention_name in masked_layers_content[before_key]:
                            masked_key = "content"
                    else:
                        # 若两个都不匹配，两边都选
                        masked_key = None
                    assert hasattr(lora_layer, "get_unziplora_cone"), lora_layer
                    # * store cone in self.attr
                    # * if it is accumulating: added to attr
                    # * if not accumulate or prepare to "determine mask": compute cone then store the "mask"
                    
                    #  计算 cone 的值，或者是将 cone 转化为列稀疏度分数
                    lora_layer.get_unziplora_cone("style", accumulate=accumulate)
                    lora_layer.get_unziplora_cone("content", accumulate=accumulate)
                    
                    if accumulate is False:
                        # 启用 column seperation
                        lora_layer.set_layer_mask("content", value=True) 
                        lora_layer.set_layer_mask("style", value=True)
                        
                        # 更新 mask
                        lora_layer.mask_updated_elements(masked_key, step_ratio=column_ratio, avoid=avoid)

                        lora_layer.set_merger_gradient("content", value=True)
                        lora_layer.set_merger_gradient("style", value=True)
                        
                        if logged and name in logged_layers_key:
                            # 返回被选中的列的cone分数
                            logged_cone_layer_content[name] = lora_layer.log_selected_mask("content")
                            logged_cone_layer_style[name] = lora_layer.log_selected_mask("style")
                        
    if logged:
        return unet, draw_concatenated_heatmap(logged_cone_layer_content), draw_concatenated_heatmap(logged_cone_layer_style)
    else:
        return unet, logged_cone_layer_content, logged_cone_layer_style
    
def lora_gradient_zeroout(unet: UNet2DConditionModel, finetune_mask):
    '''
    mask out the gradient during training to only allow training of certain columns
    '''
    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                lora_layer.set_gradient_mask(finetune_mask)


# ---------------------------生成热力图的函数和工具类---------------------------

class RecordingCrossAttnProcessor:
    def __init__(self, inner, token_ids, map_size=64):
        self.inner = inner
        self.token_ids = sorted(set(int(i) for i in token_ids))
        self.counts = {}
        self.sums = {}
        self.map_size = map_size
    
    def _accum(self, attn, hidden_states, encoder_hidden_states, encoder_hidden_states_content, 
               encoder_hidden_states_style,attention_mask):
        if (encoder_hidden_states is None and encoder_hidden_states_content is None) or not self.token_ids:
            return
        if hidden_states.dim() != 3: 
            return
        
        bsz, q_len, _ = hidden_states.shape
        attn_mask = attn.prepare_attention_mask(attention_mask, q_len, bsz)

        enc = encoder_hidden_states
        enc_content = encoder_hidden_states_content
        enc_style = encoder_hidden_states_style
        if attn.norm_cross is not None:
            enc = attn.norm_cross(enc)
            enc_content = attn.norm_cross(enc_content)
            enc_style = attn.norm_cross(enc_style)
        
        # 这里的输入是不是要和重构的 Attention Block 的输入对齐
        q, k = None, None
        if isinstance(attn.to_q.lora_layer, (UnZipLoRALinearLayer, UnZipLoRALinearLayerInfer)):
            q = attn.head_to_batch_dim(attn.to_q(hidden_states, hidden_states_1=hidden_states, hidden_states_2=hidden_states))
        else:
            q = attn.head_to_batch_dim(attn.to_q(hidden_states))
        
        if isinstance(attn.to_k.lora_layer, (UnZipLoRALinearLayer, UnZipLoRALinearLayerInfer)):
            k = attn.head_to_batch_dim(attn.to_k(enc, hidden_states_1=enc_content, hidden_states_2=enc_style))
        else:
            k = attn.head_to_batch_dim(attn.to_k(enc))

        
        # 计算注意力概率形状
        probs = attn.get_attention_scores(q, k, attn_mask)
        
        heads = attn.heads
        probs = probs.view(bsz, heads, probs.shape[-2], probs.shape[-1])

        for tid in self.token_ids:
            if tid < 0 or tid >= probs.shape[-1]:
                continue
                
            m = probs[..., tid].mean(dim=1)

            side = int(math.sqrt(m.shape[-1]))
            if side * side != m.shape[-1]:
                continue
                
            m = m.view(bsz, 1, side, side)
            m = F.interpolate(m, size=(self.map_size, self.map_size), mode="bilinear", align_corners=False)
            
            # 对于 CFG 来说，这里是在平均 CFG 的分支
            m = m.mean(dim=0).squeeze(0).detach()  # (H,W)
            if tid not in self.sums:
                self.sums[tid] = torch.zeros_like(m)
                self.counts[tid] = 0
            self.sums[tid] += m
            self.counts[tid] += 1
    
    @torch.no_grad()
    def __call__(self, attn,
                  hidden_states: torch.FloatTensor, 
                  encoder_hidden_states: Optional[torch.FloatTensor] = None, 
                  encoder_hidden_states_content: Optional[torch.FloatTensor] = None,
                  encoder_hidden_states_style: Optional[torch.FloatTensor] = None,
                  attention_mask: Optional[torch.FloatTensor] = None,
                  temb: Optional[torch.FloatTensor] = None, scale: float=1.0, *args, **kwargs):
        self._accum(attn, hidden_states, encoder_hidden_states, encoder_hidden_states_content, 
                    encoder_hidden_states_style, attention_mask)
        
        from unziplora_unet import attention_processor as uzap

        if isinstance(self.inner, (uzap.AttnProcessor, uzap.AttnProcessor2_0)):
            return self.inner(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_content=encoder_hidden_states_content,
                encoder_hidden_states_style=encoder_hidden_states_style,
                attention_mask=attention_mask,
                temb=temb, scale=scale
            )
        else:
            return self.inner(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb, scale=scale
            )
    
    def get_map(self, token_ids):
        maps = []
        for tid in token_ids:
            if tid in self.sums and self.counts[tid] > 0:
                maps.append(self.sums[tid] / self.counts[tid])
        
        if not maps:
            return None
        return torch.stack(maps, dim=0).mean(dim=0)
    
    # 新增的一个清除每张图统计数据
    def reset(self):
        self.counts.clear()
        self.sums.clear()


def attach_recorders_to_unet(unet, token_ids, map_size=64):
    wrappers, original = [], {}
    for name, module in unet.named_modules():
        if hasattr(module, "processor") and ".attn2" in name:
            original[name] = module.processor
            w = RecordingCrossAttnProcessor(module.processor, token_ids, map_size=map_size)
            module.processor = w
            wrappers.append(w)

    return wrappers, original


def aggregate_maps(recorders, token_ids):
    acc, n = None, 0
    for r in recorders:
        m = r.get_map(token_ids)
        if m is None:
            continue
        acc = m if acc is None else (acc + m)
        n += 1
    return None if n == 0 else acc / n

def overlay_heatmap(img, heat, alpha=0.45):
    heat = heat.float()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    heat_np = heat.detach().cpu().numpy()
    heat_rgb = colormaps.get_cmap("jet")(heat_np)[..., :3]
    heat_img = Image.fromarray((heat_rgb * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)

    base = np.asarray(img).astype(np.float32) / 255.0
    over = np.asarray(heat_img).astype(np.float32) / 255
    out = (1 - alpha) * base + alpha * over
    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))

def build_token_masks(tokenizer, prompts, rare_word, super_word, device):
    enc = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc.input_ids.to(device)

    rare_ids = tokenizer.encode(" " + rare_word, add_special_tokens=False)
    super_ids = tokenizer.encode(" " + super_word, add_special_tokens=False)
    rare_ids = torch.tensor(rare_ids, device=device)
    super_ids = torch.tensor(super_ids, device=device)

    B, T = input_ids.shape
    rare_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    super_mask = torch.zeros_like(rare_mask)

    for b in range(B):
        for j in range(T - len(rare_ids) + 1):
            if (rare_ids == input_ids[b, j: j + len(rare_ids)]).all():
                rare_mask[b, j: j + len(rare_ids)] = True
        
        for j in range(T - len(super_ids) + 1):
            if (super_ids == input_ids[b, j : j + len(super_ids)]).all():
                super_mask[b, j: j + len(super_ids)] = True
    return rare_mask, super_mask

def concat_triplet(a, b, c):
    w, h = a.size
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (w, 0))
    canvas.paste(c, (w * 2, 0))
    return canvas
# ---------------------------生成热力图的函数和工具类---------------------------

