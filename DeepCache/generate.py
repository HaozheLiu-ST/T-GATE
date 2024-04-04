import torch
from sdxl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline as DeepCacheStableDiffusionXLPipeline

model = "stabilityai/stable-diffusion-xl-base-1.0"
gate_idx = 4
cache_interval = 3
cache_layer_id = 0
cache_block_id = 0
num_inference_steps = 25
pipe = DeepCacheStableDiffusionXLPipeline.from_pretrained(
    model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = prompt = "A haunted Victorian mansion under a full moon."
generator = torch.manual_seed('10086')
deepcache_output = pipe(
    prompt, 
    num_inference_steps=num_inference_steps,
    cache_interval=cache_interval, cache_layer_id=cache_layer_id, cache_block_id=cache_block_id,
    uniform=True, 
    generator=generator,
    gate_idx = gate_idx,
    return_dict=True
).images[0]
deepcache_output.save('ours.png')
