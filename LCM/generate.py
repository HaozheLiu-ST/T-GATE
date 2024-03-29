from diffusers import UNet2DConditionModel, LCMScheduler
from diffusers import StableDiffusionXLPipeline
import torch
from customize_attention import register_custom_forward




unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl",
    torch_dtype=torch.float16,
    variant="fp16",
)

gate_step = 1
inference_num_per_image = 4

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16",
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

register_custom_forward(pipe.unet, 'Attention',gate_step=gate_step,inference_num_per_image =inference_num_per_image)

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
generator = torch.manual_seed(0)
image = pipe(
    prompt=prompt, num_inference_steps=inference_num_per_image, generator=generator
).images[0]

image.save('test.png')
