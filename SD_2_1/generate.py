from diffusers import DPMSolverMultistepScheduler
import torch
from stable_diffusion import StableDiffusionPipeline 
import pickle
from customize_attention import register_custom_forward

repo_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
inference_num_per_image = 25
gate_step = 10

register_custom_forward(pipe.unet, 'Attention',gate_step=gate_step,inference_num_per_image =inference_num_per_image)

pipe = pipe.to("cuda")

prompt = "High quality photo of an astronaut riding a horse in space"

generator = torch.manual_seed('10086')

img = pipe.tgate(prompt, 
                num_inference_steps=inference_num_per_image, 
                guidance_scale=7.5, 
                gate_step=gate_step, 
                genertaor = generator).images[0]


img.save('test.png')
