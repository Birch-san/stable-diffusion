import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from torch import Tensor
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
# from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast, nn
from contextlib import contextmanager, nullcontext
from random import randint

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from k_diffusion.sampling import sample_lms, sample_dpm_2, sample_dpm_2_ancestral, sample_euler, sample_euler_ancestral, sample_heun, get_sigmas_karras, make_dynamic_thresholder
from k_diffusion.external import CompVisDenoiser

def get_device():
    if(torch.cuda.is_available()):
        return 'cuda'
    elif(torch.backends.mps.is_available()):
        return 'mps'
    else:
        return 'cpu'

# samplers from the Karras et al paper
KARRAS_SAMPLERS = { 'heun', 'euler', 'dpm2' }
K_DIFF_SAMPLERS = { *KARRAS_SAMPLERS, 'k_lms', 'dpm2_ancestral', 'euler_ancestral' }
NOT_K_DIFF_SAMPLERS = { 'ddim', 'plms' }
VALID_SAMPLERS = { *K_DIFF_SAMPLERS, *NOT_K_DIFF_SAMPLERS }

class KCFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(get_device())
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

def check_safety_poorly(images, **kwargs):
    return images, False

# https://github.com/lucidrains/imagen-pytorch/blob/ceb23d62ecf611082c82b94f2625d78084738ced/imagen_pytorch/imagen_pytorch.py#L127
# from lucidrains' imagen_pytorch
# MIT-licensed
def right_pad_dims_to(x: Tensor, t: Tensor) -> Tensor:
  padding_dims = x.ndim - t.ndim
  if padding_dims <= 0:
    return t
  return t.view(*t.shape, *((1,) * padding_dims))

def main():
    parser = argparse.ArgumentParser()

    proposed_seed = randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of sampling steps",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        help="which sampler",
        choices=VALID_SAMPLERS,
        default="k_lms"
    )
    parser.add_argument(
        "--karras",
        action='store_true',
        help=f"use Karras et al noise scheduler (higher FID / fewer steps). only {KARRAS_SAMPLERS} samplers are supported. (i.e. can quantize sigma_hat to discrete noise levels). you can *try* the Karras et al noise scheduler with other samplers if you want, but it will probably be sub-optimal (as you will lack the mitigations described in section C.3.4 of the arXiv:2206.00364 paper).",
    )
    parser.add_argument(
        "--dynamic_thresholding",
        action='store_true',
    )
    parser.add_argument(
        "--dynamic_thresholding_percentile",
        type=float,
        default=0.9,
        help="implements dynamic thresholding from Imagen paper, section E Figure A.32. pushes saturated pixels (those near -1 and 1) inwards at every step, preventing saturation"
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=proposed_seed,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device(get_device())
    model = model.to(device)

    if opt.sampler in K_DIFF_SAMPLERS:
        model_k_wrapped = CompVisDenoiser(model)
        model_k_config = KCFGDenoiser(model_k_wrapped)
    elif opt.sampler in NOT_K_DIFF_SAMPLERS:
        if opt.sampler == 'plms':
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)
    

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    # wm_encoder = WatermarkEncoder()
    # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

    start_code = None

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    if device.type == 'mps':
        precision_scope = nullcontext # have to use f32 on mps
    with torch.no_grad():
        with precision_scope(device.type):
            with model.ema_scope():
                tic = time.perf_counter()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    iter_tic = time.perf_counter()
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        if start_code is None or not opt.fixed_code:
                            rand_size = [opt.n_samples, *shape]
                            # https://github.com/CompVis/stable-diffusion/issues/25#issuecomment-1229706811
                            # MPS random is not currently deterministic w.r.t seed, so compute randn() on-CPU
                            start_code = torch.randn(rand_size, device='cpu').to(device) if device.type == 'mps' else torch.randn(rand_size, device=device)

                        if opt.sampler in NOT_K_DIFF_SAMPLERS:
                            samples, _ = sampler.sample(S=opt.steps,
                                                            conditioning=c,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code)
                        elif opt.sampler in K_DIFF_SAMPLERS:
                            match opt.sampler:
                                case 'dpm2':
                                    sampling_fn = sample_dpm_2
                                case 'dpm2_ancestral':
                                    sampling_fn = sample_dpm_2_ancestral
                                case 'heun':
                                    sampling_fn = sample_heun
                                case 'euler':
                                    sampling_fn = sample_euler
                                case 'euler_ancestral':
                                    sampling_fn = sample_euler_ancestral
                                case 'k_lms' | _:
                                    sampling_fn = sample_lms

                            noise_schedule_sampler_args = {}
                            # Karras sampling schedule achieves higher FID in fewer steps
                            # https://arxiv.org/abs/2206.00364
                            if opt.karras:
                                sigmas = get_sigmas_karras(
                                    # ordinarily the step count is inclusive of the zero we're given, but we're excluding zero, 
                                    n=opt.steps+1,
                                    # 0.0292
                                    sigma_min=model_k_wrapped.sigmas[0].item(),
                                    # 14.6146
                                    sigma_max=model_k_wrapped.sigmas[-1].item(),
                                    rho=7.,
                                    device=device,
                                    # zero would be smaller than sigma_min
                                    concat_zero=False
                                )
                                if opt.sampler in KARRAS_SAMPLERS:
                                    noise_schedule_sampler_args['quanta'] = model_k_wrapped.sigmas
                                else:
                                    print(f'[WARN] you have requested a Karras et al noise schedule, but "{opt.sampler}" sampler does not implement time step discretization (as described in section C.3.4 of arXiv:2206.00364). we will discretize the sigmas before we give them to the sampler. maybe that suffices. if wrong, then it will be suboptimal (and therefore you may not get the benefits you were hoping for). prefer a sampler from {KARRAS_SAMPLERS}.')
                                    # quantize sigmas from noise schedule to closest equivalent in model_k_wrapped.sigmas
                                    sigmas = model_k_wrapped.sigmas[torch.argmin((sigmas.reshape(len(sigmas), 1).repeat(1, len(model_k_wrapped.sigmas)) - model_k_wrapped.sigmas).abs(), dim=1)]
                            else:
                                sigmas = model_k_wrapped.get_sigmas(opt.steps)
                            
                            postprocess_step = make_dynamic_thresholder(opt.dynamic_thresholding_percentile) if opt.dynamic_thresholding else None

                            x = start_code * sigmas[0] # for GPU draw
                            extra_args = {
                                'cond': c,
                                'uncond': uc,
                                'cond_scale': opt.scale,
                            }
                            samples = sampling_fn(
                                model_k_config,
                                x,
                                sigmas,
                                extra_args=extra_args,
                                postprocess_step=postprocess_step,
                                **noise_schedule_sampler_args)

                        x_samples = model.decode_first_stage(samples)
                        
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image, has_nsfw_concept = check_safety_poorly(x_samples)

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        if not opt.skip_save:
                            for ix, x_sample in enumerate(x_checked_image_torch):
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                # img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(sample_path, f"{base_count:05}.s{opt.seed}.n{n}.i{ix}.png"))
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_checked_image_torch)
                    iter_toc = time.perf_counter()
                    print(f'batch {n} generated {batch_size} images in {iter_toc-iter_tic} seconds')
                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    # img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(outpath, f'grid-{grid_count:04}.s{opt.seed}.png'))
                    grid_count += 1

                toc = time.perf_counter()
                print(f'in total, generated {opt.n_iter} batches of {batch_size} images in {toc-tic} seconds')

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
