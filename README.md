# TGATE

[TGATE](https://github.com/HaozheLiu-ST/T-GATE/tree/main) accelerates inferences of [`PixArtAlphaPipeline`], [`PixArtSigmaPipeline`], [`StableDiffusionPipeline`], [`StableDiffusionXLPipeline`], and [`StableVideoDiffusionPipeline`] by skipping the calculation of self-attention and cross-attention once it converges. More details can be found at [technical report](https://huggingface.co/papers/2404.02747).

![](https://github.com/user-attachments/assets/44805d66-e504-4de4-837d-d027fb3f566b)


## 🚀 Major Features

* Training-Free.
* Easily Integrate into [Diffusers](https://github.com/huggingface/diffusers/tree/main).
* Only a few lines of code are required.
* Complementary to [DeepCache](https://github.com/horseee/DeepCache).
* Friendly support [Stable Diffusion pipelines](https://huggingface.co/stabilityai), [PixArt](https://pixart-alpha.github.io/), and [Latent Consistency Models](https://latent-consistency-models.github.io/).
* 10%-50% speed up for different models. 

## 📖 Quick Start

### 🛠️ Installation

Start by installing [TGATE](https://github.com/HaozheLiu-ST/T-GATE/tree/releases):

```bash
pip install tgate
```

#### Requirements

* pytorch>=2.0.0
* diffusers>=0.29.0
* DeepCache==0.1.1
* transformers
* accelerate

### 🌟 Usage

Accelerate `PixArtAlphaPipeline` with TGATE:

```diff
import torch
from diffusers import PixArtAlphaPipeline

pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS", 
    torch_dtype=torch.float16,
)

+ from tgate import TgatePixArtAlphaLoader
+ gate_step = 15
+ sp_interval = 3
+ fi_interval = 1
+ warm_up = 2
+ inference_step = 25
+ pipe = TgatePixArtAlphaLoader(pipe).to("cuda")

+ image = pipe.tgate(
+         "An alpaca made of colorful building blocks, cyberpunk.",
+         gate_step=gate_step,
+         sp_interval=sp_interval,
+         fi_interval=fi_interval,
+         warm_up=warm_up,   
+         num_inference_steps=inference_step,
+ ).images[0]
```

Accelerate `PixArtSigmaPipeline` with TGATE:

```diff
import torch
from diffusers import PixArtSigmaPipeline

pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", 
    torch_dtype=torch.float16,
)

+ from tgate import TgatePixArtSigmaLoader
+ gate_step = 15
+ sp_interval = 3
+ fi_interval = 1
+ warm_up = 2
+ inference_step = 25
+ pipe = TgatePixArtSigmaLoader(pipe).to("cuda")

+ image = pipe.tgate(
+         "an astronaut sitting in a diner, eating fries, cinematic, analog film.",
+         gate_step=gate_step,
+         sp_interval=sp_interval,
+         fi_interval=fi_interval,
+         warm_up=warm_up,   
+         num_inference_steps=inference_step,
+ ).images[0]
```

Accelerate `StableDiffusionXLPipeline` with TGATE:

```diff
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)

+ from tgate import TgateSDXLLoader
+ gate_step = 10
+ sp_interval = 5
+ fi_interval = 1
+ warm_up = 2
+ inference_step = 25
+ pipe = TgateSDXLLoader(pipe).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

+ image = pipe.tgate(
+         "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k.",
+         gate_step=gate_step,
+         sp_interval=sp_interval,
+         fi_interval=fi_interval,
+         warm_up=warm_up,  
+         num_inference_steps=inference_step
+ ).images[0]
```

Accelerate `StableDiffusionXLPipeline` with [DeepCache](https://github.com/horseee/DeepCache) and TGATE:

```diff
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)

+ from tgate import TgateSDXLDeepCacheLoader
+ gate_step = 10
+ sp_interval = 1
+ fi_interval = 1
+ warm_up = 0
+ inference_step = 25
+ pipe = TgateSDXLDeepCacheLoader(
+        pipe,
+        cache_interval=3,
+        cache_branch_id=0,
+ ).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


+ image = pipe.tgate(
+         "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k.",
+         gate_step=gate_step,
+         sp_interval=sp_interval,
+         fi_interval=fi_interval,
+         warm_up=warm_up,  
+         num_inference_steps=inference_step
+ ).images[0]
```

Accelerate `latent-consistency/lcm-sdxl` with TGATE:

```diff
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import UNet2DConditionModel, LCMScheduler
from diffusers import DPMSolverMultistepScheduler

unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    unet=unet,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

+ from tgate import TgateSDXLLoader
+ gate_step = 1
+ sp_interval = 1
+ fi_interval = 1
+ warm_up = 0
+ inference_step = 4
+ pipe = TgateSDXLLoader(pipe,lcm=True).to("cuda")

+ image = pipe.tgate(
+         "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k.",
+         gate_step=gate_step,
+         sp_interval=sp_interval,
+         fi_interval=fi_interval,
+         warm_up=warm_up,  
+         num_inference_steps=inference_step,
+ ).images[0]
```

TGATE also supports `StableDiffusionPipeline`, `PixArt-alpha/PixArt-LCM-XL-2-1024-MS`, and `StableVideoDiffusionPipeline`.
More details can be found at [here](https://github.com/HaozheLiu-ST/T-GATE/tree/releases/main.py).

## 📄 Results
| Model                 | MACs     | Latency | Zero-shot 10K-FID on MS-COCO |
|-----------------------|----------|---------|---------------------------|
| SD-XL                 | 149.438T | 53.187s | 24.164                    |
| SD-XL w/ TGATE        | 95.988T  | 31.643s | 22.917                    |
| Pixart-Alpha          | 107.031T | 61.502s | 37.983                    |
| Pixart-Alpha w/ TGATE | 73.971T  | 36.650s | 36.390                    |
| Pixart-Sigma          | 107.766T | 60.467s | 34.278                    |
| Pixart-Sigma w/ TGATE | 74.420T  | 36.449s | 32.927                    |
| DeepCache (SD-XL)     | 57.888T  | 19.931s | 25.678                    |
| DeepCache w/ TGATE    | 43.868T  | 14.666s | 24.511                    |
| LCM (SD-XL)           | 11.955T  | 3.805s  | 26.357                    |
| LCM w/ TGATE          | 11.171T  | 3.533s  | 26.902                    |
| LCM (Pixart-Alpha)    | 8.563T   | 4.733s  | 35.989                    |
| LCM w/ TGATE          | 7.623T   | 4.543s  | 35.843                    |

The FID is computed on [captions](https://github.com/HaozheLiu-ST/T-GATE/files/15369063/idx_caption.txt) by [PytorchFID](https://github.com/mseitzer/pytorch-fid).

The latency is tested on a 1080ti commercial card and diffusers [v0.28.2](https://github.com/huggingface/diffusers/tree/v0.28.2). 

The MACs and Params are calculated by [calflops](https://github.com/MrYxJ/calculate-flops.pytorch). 

## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
```
@article{tgate_v2,
  title={Faster Diffusion via Temporal Attention Decomposition},
  author={Liu, Haozhe and Zhang, Wentian and Xie, Jinheng and Faccio, Francesco and Xu, Mengmeng and Xiang, Tao and Shou, Mike Zheng and Perez-Rua, Juan-Manuel and Schmidhuber, J{\"u}rgen},
  journal={arXiv preprint arXiv:2404.02747},
  year={2024}
}
```
