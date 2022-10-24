import os

import torch
from diffusers import StableDiffusionPipeline, KarrasVeScheduler
# from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from PIL import Image
import time

kve = KarrasVeScheduler(
    sigma_max=14.6146,
    # sigma_min=0.0936,
    sigma_min=0.0292,
    s_churn=0.
)
# lms = LMSDiscreteScheduler(
#   beta_start=0.00085,
#   beta_end=0.012,
#   beta_schedule="scaled_linear"
# )

pipe = StableDiffusionPipeline.from_pretrained("/Users/birch/git/stable-diffusion-v1-4", safety_checker=None)# torch_type=torch.float16, revision="fp16")

tic = time.perf_counter()
pipe = pipe.to("mps")

prompt = "masterpiece character portrait of a blonde girl, full resolution, 4k, mizuryuu kei, akihiko. yoshida, Pixiv featured, baroque scenic, by artgerm, sylvain sarrailh, rossdraws, wlop, global illumination, vaporwave"
generator = torch.Generator(device="cpu").manual_seed(68673924)
image: Image.Image = pipe(
	prompt,
	# guidance_scale=1.,
	generator=generator,  
  # scheduler=lms,
  scheduler=kve,
  # num_inference_steps=30
  num_inference_steps=15
).images[0]

sample_path="outputs/diffusers"
base_count = len(os.listdir(sample_path))
image.save(os.path.join(sample_path, f"{base_count:05}.png"))
toc = time.perf_counter()
print(f'in total, generated 1 image in {toc-tic} seconds')