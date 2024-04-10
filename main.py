import argparse
import os

import torch
from src.tgate import TgateSDXLLoader,TgatePixArtLoader,TgateSDLoader,TgateSDDeepCacheLoader,TgateSDXLDeepCacheLoader
from diffusers import PixArtAlphaPipeline,StableDiffusionXLPipeline,StableDiffusionPipeline
from diffusers import UNet2DConditionModel, LCMScheduler
from diffusers import DPMSolverMultistepScheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of TGATE.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="the input prompts",
    )
    parser.add_argument(
        "--saved_path",
        type=str,
        default=None,
        required=True,
        help="Path to save the generated results.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='pixart',
        help="[pixart,sd_xl,sd_2.1,sd_1.5,lcm_sdxl,lcm_pixart]",
    )
    parser.add_argument(
        "--gate_step",
        type=int,
        default=10,
        help="When re-using the cross-attention",
    )

    parser.add_argument(
        "--inference_step",
        type=int,
        default=25,
        help="total inference steps",

    parser.add_argument(
        '--deepcache', 
        action='store_true', 
        default=False, 
        help='do deep cache'
        )
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.saved_path, exist_ok=True)
    saved_path = os.path.join(args.saved_path, 'test.png')
    if args.model in ['sd_2.1', 'sd_1.5']:
        if args.model == 'sd_1.5':
            repo_id = "runwayml/stable-diffusion-v1-5"
        elif args.model == 'sd_2.1':
            repo_id = "stabilityai/stable-diffusion-2-1"

        pipe = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        if args.deepcache:
            pipe = TgateSDDeepCacheLoader(pipe,cache_interval=3,cache_branch_id=0)
        else:
            pipe = TgateSDLoader(pipe,gate_step=args.gate_step, num_inference_steps=args.inference_step)
        pipe = pipe.to("cuda")

        image = pipe.tgate(args.prompt,
                        num_inference_steps=args.inference_step,
                        guidance_scale=7.5,
                        gate_step=args.gate_step,
                        ).images[0]


    elif args.model == 'sd_xl':
        pipeline_text2image = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )

        if args.deepcache:
            pipeline_text2image = TgateSDXLDeepCacheLoader(pipeline_text2image,cache_interval=3,cache_branch_id=0)
        else:
            pipeline_text2image = TgateSDXLLoader(pipeline_text2image,gate_step=args.gate_step, num_inference_steps=args.inference_step)
        pipeline_text2image.scheduler = DPMSolverMultistepScheduler.from_config(pipeline_text2image.scheduler.config)
        pipeline_text2image = pipeline_text2image.to("cuda")

        image = pipeline_text2image.tgate(prompt=args.prompt,
                                          gate_step=args.gate_step,
                                          num_inference_steps=args.inference_step).images[0]
    elif args.model == 'pixart':
        pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)
        pipe = TgatePixArtLoader(pipe,gate_step=args.gate_step, num_inference_steps=args.inference_step,lcm=False)

        pipe = pipe.to("cuda")
        image = pipe.tgate(args.prompt,
                           gate_step=args.gate_step,
                           num_inference_steps=args.inference_step).images[0]


    elif args.model == 'lcm_pixart':
        pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-LCM-XL-2-1024-MS", torch_dtype=torch.float16)
        pipe = TgatePixArtLoader(pipe,gate_step=args.gate_step, num_inference_steps=args.inference_step,lcm=True)
        pipe = pipe.to("cuda")
        image = pipe(args.prompt,
                     gate_step=args.gate_step,
                     num_inference_steps=args.inference_step,
                     guidance_scale=0.).images[0]


    elif args.model == 'lcm_sdxl':
        unet = UNet2DConditionModel.from_pretrained(
            "latent-consistency/lcm-sdxl",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16",
        )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe = TgateSDXLLoader(pipe,gate_step=args.gate_step, num_inference_steps=args.inference_step,lcm=True)
        pipe = pipe.to("cuda")

        image = pipe(
            prompt=args.prompt, num_inference_steps=args.inference_step
        ).images[0]

    else:
        raise Exception('Please sepcify the model name!')
    image.save(saved_path)
