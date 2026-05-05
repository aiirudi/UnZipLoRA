import argparse
import torch 
import os

from typing import Optional

from diffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
)

from unziplora_unet.utils import *
from unziplora_unet.pipeline_stable_diffusion_xl import StableDiffusionXLSingleLoRAPipeline

MODEL_ID = "/home/xzh/xzh/pretrained/sd_xl_base_1.0"
# MODEL_ID="etri-vilab/koala-lightning-1b"
seeds = [0, 1000, 111, 1234]
device = "cuda" if torch.cuda.is_available() else "cpu"
weight_dtype = torch.float16

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def build_inference_mask(args, pipe, rare_token=None, device="cpu"):
    prompts = [args.prompt]  
    def _mask_for_tokenizer(tokenizer, rare_token):
                                                             
        if rare_token is not None:
            mask = build_token_masks(
                tokenizer, 
                prompts,
                rare_word=rare_token,
                device=device,
            )
        return mask

    if args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-v1-5":
        mask = _mask_for_tokenizer(pipe.tokenizer, rare_token)
        mask = mask.unsqueeze(-1)  
        return mask
    else:
        mask1 = _mask_for_tokenizer(pipe.tokenizer, rare_token)
        mask2 = _mask_for_tokenizer(pipe.tokenizer_2, rare_token)
        mask = mask1 | mask2
        mask = mask.unsqueeze(-1)
        return mask

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--with_unziplora",
        action="store_true",
        help="Whether use different prompts to generate",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=4,
        help=("The number of generated figures of each seed."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help=("The directory for saved model"),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="example_output",
        help=("The directory for saved generated figures"),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help=("The prompt for validation"),
    )

    parser.add_argument(
        "--validation_prompt_content_forward",
        type=str,
        default=None,
        help=("The prompt for validation"),
    )
    parser.add_argument(
        "--validation_prompt_style_forward",
        type=str,
        default=None,
        help=("The prompt for validation"),
    )

    parser.add_argument(
        "--validation_prompt_content_recontext",
        type=str,
        default=None,
        help=("The content recontext prompt for validation"),
    )

    parser.add_argument(
        "--validation_prompt_style",
        type=str,
        default=None,
        help=("The style prompt for validation"),
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    
    # TFM rare token 参数
    parser.add_argument(
        "--focus_value",
        type=str2bool,
        default="false",
        help=("是否启用 TFM mask 的值"),
    )
    parser.add_argument(
        "--content_rare_word",
        type=str,
        default="",
        help=("rare token used for inference TFM mask"),
    )


    # 随机矩阵svd初始化
    parser.add_argument(
        "--use_base_weight",
        type=str2bool,
        default="true",
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument("--with_svd_init",type=str2bool, default="true", help="是否使用随机矩阵svd分解初始化权重;关闭是使用nn.Linear默认初始化")
    
    # 加入content LoRA 和 style LoRA 的非对称时间步策略
    parser.add_argument("--min_rank_content",type=int,default=32,help="content LoRA非堆成时间步的最小秩数") 
    parser.add_argument("--min_rank_style",type=int,default=24,help="style LoRA非堆成时间步的最小秩数") 
    parser.add_argument("--alpha", type=float, default=1.0, help='时间步系数控制参数')
    parser.add_argument("--timestep_mode", type=str,choices=["priecewise", "linear"],default="priecewise", help='时间步系数计算模式')
    parser.add_argument("--use_time_control", type=str2bool, default='false', help='启用时间步mask')
    
    args = parser.parse_args()
    
    return args 

def log_validation(pipeline, prompt, prompt_content="", prompt_style="", content_rare_word: Optional[str]=None,seed=0, num=4):
    generator = torch.Generator(device=device).manual_seed(seed)
    # Currently the context determination is a bit hand-wavy. We can improve it in the future if there's a better
    # way to condition it. Reference: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
    if pipeline.__class__.__name__ == 'StableDiffusionXLUnZipLoRAPipeline':
        pipeline_args = {"prompt": prompt, 
                        "prompt_content": prompt_content, 
                        "prompt_style": prompt_style,
                        "content_rare_word":content_rare_word}
    else: 
        pipeline_args = {"prompt": prompt,
                        "content_rare_word":content_rare_word}
        
    images = [pipeline(**pipeline_args, generator=generator, num_inference_steps=50).images[0] for _ in range(num)]
    return images

def save_img(img_dir, images, img_num):
    for _, img in enumerate(images):
        image_path = os.path.join(img_dir, f"image_{img_num}.png")
        img_num += 1
        img.save(image_path, "PNG")
    return img_num

def generate_save_img(args, pipeline, prompt, prompt_catogory, prompt_content_forward=None, prompt_style_forward=None):
    for i in range(len(prompt)):
        img_num = 1
        prompt_dir = os.path.join(prompt_catogory, "_".join(prompt[i].split(" ")))
        if os.path.isdir(prompt_dir):
            continue
        os.makedirs(prompt_dir, exist_ok=True)
        for seed in seeds:
            print(prompt[i])
            if pipeline.__class__.__name__ == 'StableDiffusionXLUnZipLoRAPipeline':
                images = log_validation(
                    pipeline,
                    prompt[i],
                    prompt_content_forward[i],
                    prompt_style_forward[i],
                    content_rare_word=args.content_rare_word if args.focus_value else None,
                    seed = seed, 
                    num = args.num
                )
            else:
                images = log_validation(
                    pipeline,
                    prompt[i],
                    seed = seed, 
                    num = args.num,
                    content_rare_word=args.content_rare_word if args.focus_value else None,
                )
            img_num = save_img(prompt_dir, images, img_num)

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    with torch.no_grad():
        
        # * Generate combined images
        vae = AutoencoderKL.from_pretrained( #载入vae
            MODEL_ID,
            subfolder="vae",
            revision=None
        )
        print("model dir: ", args.output_dir)
        
        # 说明这段是 combine content and style
        if len(args.validation_prompt) != 0 and args.validation_prompt != ['']: 
            # 加载如 SDXL-UnZipLoRA-Pipline， 能同时接受三种 prompt
            pipeline = load_pipeline_from_sdxl(
            MODEL_ID, vae = vae,
            max_rank=args.rank,
            min_rank_content=args.min_rank_content,
            min_rank_style=args.min_rank_style,
            alpha=args.alpha,
            timestep_mode=args.timestep_mode,
            use_time_control=args.use_time_control,
            use_base_weight=args.use_base_weight,
            )
            if args.with_unziplora:
                pipeline.unet = insert_unziplora_to_unet(pipeline.unet, 
                    f"{args.output_dir}_content", 
                    f"{args.output_dir}_style",
                    weight_content_path=f"{args.output_dir}_merger_content.pth",
                    weight_style_path=f"{args.output_dir}_merger_style.pth",
                    base_weight_content_path=f"{args.output_dir}_base_weight_content.pth",
                    base_weight_style_path=f"{args.output_dir}_base_weight_style.pth",
                    use_base_weight=args.use_base_weight,
                    rank=args.rank)
            else:
                pipeline.unet = insert_unziplora_to_unet(pipeline.unet, 
                    f"{args.output_dir}_content", 
                    f"{args.output_dir}_style",
                    base_weight_content_path=f"{args.output_dir}_base_weight_content.pth",
                    base_weight_style_path=f"{args.output_dir}_base_weight_style.pth",
                    rank=args.rank)
            pipeline = pipeline.to(device, dtype=weight_dtype)
            prompt_catogory = os.path.join(args.save_dir, "combine_recontextual_outputs")
            os.makedirs(prompt_catogory, exist_ok=True)
            
            # 在推理过程中为 content prompt 分支加上 TFM


            if args.with_unziplora:
                generate_save_img(args, pipeline, args.validation_prompt, prompt_catogory, \
                    args.validation_prompt_content_forward, 
                    args.validation_prompt_style_forward)
            else:
                generate_save_img(args, pipeline, args.validation_prompt, prompt_catogory)
            
            del pipeline
        
        # 只生成 content 图片
        if len(args.validation_prompt_content_recontext) != 0 and args.validation_prompt_content_recontext != ['']: 
            prompt_catogory = os.path.join(args.save_dir, "content_recontextual_outputs")
            os.makedirs(prompt_catogory, exist_ok=True)
            
            if args.with_svd_init:
                pipeline = StableDiffusionXLSingleLoRAPipeline.from_pretrained(
                    MODEL_ID,
                    vae=vae,
                    torch_dtype=weight_dtype,
                    revision=None, 
                    max_rank=args.rank,
                    min_rank=args.min_rank_content,
                    alpha=args.alpha,
                    branch="content",
                    timestep_mode=args.timestep_mode,
                    use_time_control=args.use_time_control,
                    use_base_weight=args.use_base_weight,
                )
                pipeline.setup_lora_layers(lora_path=f"{args.output_dir}_content", rank=args.rank, use_base_weight=args.use_base_weight,
                base_weight_path=f"{args.output_dir}_base_weight_content.pth",branch="content")
            else:
                pipeline = StableDiffusionXLPipeline.from_pretrained(MODEL_ID,)
                pipeline.load_lora_weights(f"{args.output_dir}_content")
            pipeline = pipeline.to(device, dtype=weight_dtype)

            # pipeline.load_lora_weights(f"{args.output_dir}")
            print(f"generate recontext prompt {args.validation_prompt_content_recontext}")
            generate_save_img(args, pipeline, args.validation_prompt_content_recontext, prompt_catogory)
            
            del pipeline        

        # 只生成 style 图片
        if len(args.validation_prompt_style) != 0 and args.validation_prompt_style != ['']: 
            prompt_catogory = os.path.join(args.save_dir, "style_recontextual_outputs")
            os.makedirs(prompt_catogory, exist_ok=True)
            
            if args.with_svd_init:
                pipeline = StableDiffusionXLSingleLoRAPipeline.from_pretrained(
                    MODEL_ID,
                    vae=vae,
                    revision=None,
                    torch_dtype=weight_dtype,
                    max_rank=args.rank,
                    min_rank=args.min_rank_style,
                    alpha=args.alpha,
                    branch="style",
                    timestep_mode=args.timestep_mode,
                    use_time_control=args.use_time_control,
                    use_base_weight=args.use_base_weight,
                )
                pipeline.setup_lora_layers(lora_path=f"{args.output_dir}_style", rank=args.rank, use_base_weight=args.use_base_weight,
                base_weight_path=f"{args.output_dir}_base_weight_style.pth",branch="style")
            else:
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                MODEL_ID,
                )
                pipeline.load_lora_weights(f"{args.output_dir}_style")
            
            pipeline = pipeline.to(device, dtype=weight_dtype)
            # pipeline.load_lora_weights(f"{args.output_dir}")
            print(f"generate recontext prompt {args.validation_prompt_style}")
            generate_save_img(args, pipeline, args.validation_prompt_style, prompt_catogory)
        
if __name__ == "__main__":
    args = parse_args()
    args.validation_prompt=args.validation_prompt.split(",")
    args.validation_prompt_style_forward=args.validation_prompt_style_forward.split(",")
    args.validation_prompt_content_forward=args.validation_prompt_content_forward.split(",")
    args.validation_prompt_content_recontext=args.validation_prompt_content_recontext.split(",")
    args.validation_prompt_style=args.validation_prompt_style.split(",")
    
    main(args)
