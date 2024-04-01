import torch
from pipeline_pixart_alpha import PixArtAlphaPipeline
from customize_attention import register_custom_forward
# You can replace the checkpoint id with "PixArt-alpha/PixArt-XL-2-512x512" too.
gate_step = 8
inference_num_per_image = 25
pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)
register_custom_forward(pipe.transformer, 'Attention',gate_step=gate_step,inference_num_per_image =inference_num_per_image)
pipe = pipe.to("cuda")

generator = torch.manual_seed('10086')

prompt = "An alpaca made of colorful building blocks, cyberpunk."
image = pipe.tgate(prompt,gate_step=gate_step,num_inference_steps=inference_num_per_image,generator=generator).images[0]
image.save('test.png')
