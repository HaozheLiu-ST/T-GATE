from diffusers import AutoPipelineForText2Image
from pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler
import torch
from customize_attention import register_custom_forward


gate_step = 10
inference_num_per_image = 25
pipeline_text2image = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
generator = torch.manual_seed('10086')
pipeline_text2image.scheduler = DPMSolverMultistepScheduler.from_config(pipeline_text2image.scheduler.config)
register_custom_forward(pipeline_text2image.unet, 'Attention',gate_step=gate_step,inference_num_per_image =inference_num_per_image)
pipeline_text2image = pipeline_text2image.to("cuda")


prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline_text2image.tgate(prompt=prompt,gate_step=gate_step, num_inference_steps = inference_num_per_image,generator=generator).images[0]
image.save('ours.png')