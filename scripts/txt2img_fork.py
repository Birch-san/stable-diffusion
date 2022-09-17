import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from torch import Tensor, FloatTensor
from omegaconf import OmegaConf
from PIL import Image
from PIL.Image import Resampling
from tqdm import tqdm, trange
# from imwatermark import WatermarkEncoder
from itertools import islice, repeat as repeat_, chain, pairwise
from einops import rearrange, repeat
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast, nn
from contextlib import contextmanager, nullcontext
from random import randint
from typing import Generic, Optional, Iterable, List, TypeAlias, Tuple, TypeVar, Callable
import re
from ldm.models.diffusion.ddpm import LatentDiffusion
import abc
from dataclasses import dataclass
from enum import Enum

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from k_diffusion.sampling import sample_lms, sample_dpm_2, sample_dpm_2_ancestral, sample_euler, sample_euler_ancestral, sample_heun, get_sigmas_karras, append_zero
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
NON_KARRAS_K_DIFF_SAMPLERS = { 'k_lms', 'dpm2_ancestral', 'euler_ancestral' }
K_DIFF_SAMPLERS = { *KARRAS_SAMPLERS, *NON_KARRAS_K_DIFF_SAMPLERS }
NOT_K_DIFF_SAMPLERS = { 'ddim', 'plms' }
VALID_SAMPLERS = { *K_DIFF_SAMPLERS, *NOT_K_DIFF_SAMPLERS }

class KCFGDenoiser(nn.Module):
    inner_model: CompVisDenoiser
    def __init__(self, model: CompVisDenoiser):
        super().__init__()
        self.inner_model = model

    def forward(
        self,
        x: Tensor,
        sigma: Tensor,
        uncond: Tensor,
        cond: Tensor, 
        cond_scale: float,
        cond_arities: Iterable[int],
        cond_weights: Optional[Iterable[float]]
    ) -> Tensor:
        uncond_count = uncond.size(dim=0)
        cond_count = cond.size(dim=0)
        cond_in = torch.cat((uncond, cond))
        del uncond, cond
        cond_arities_tensor = torch.tensor(cond_arities, device=cond_in.device)
        x_in = cat_self_with_repeat_interleaved(t=x, factors_tensor=cond_arities_tensor, factors=cond_arities, output_size=cond_count)
        del x
        sigma_in = cat_self_with_repeat_interleaved(t=sigma, factors_tensor=cond_arities_tensor, factors=cond_arities, output_size=cond_count)
        del sigma
        uncond_out, conds_out = self.inner_model(x_in, sigma_in, cond=cond_in).split([uncond_count, cond_count])
        del x_in, sigma_in, cond_in
        unconds = repeat_interleave_along_dim_0(t=uncond_out, factors_tensor=cond_arities_tensor, factors=cond_arities, output_size=cond_count)
        del cond_arities_tensor
        # transform
        #   tensor([0.5, 0.1])
        # into:
        #   tensor([[[[0.5000]]],
        #           [[[0.1000]]]])
        weight_tensor = (torch.tensor(cond_weights, device=uncond_out.device) * cond_scale).reshape(len(cond_weights), 1, 1, 1)
        deltas: Tensor = (conds_out-unconds) * weight_tensor
        del conds_out, unconds, weight_tensor
        cond = sum_along_slices_of_dim_0(deltas, arities=cond_arities)
        del deltas
        return uncond_out + cond

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# from transformers import AutoFeatureExtractor


# load safety model
# safety_model_id = "CompVis/stable-diffusion-safety-checker"
# safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

T = TypeVar('T')

@dataclass
class InBetweenParams(Generic[T]):
    from_: T
    to: T
    step: float

MakeInbetween: TypeAlias = Callable[[InBetweenParams], T]

def intersperse_linspace(
    lst: List[T],
    make_inbetween: MakeInbetween[T],
    steps: Optional[int]
) -> List[T]:
    if steps is None:
        return lst
    return [
        *chain(
            *(
                (
                    pair[0],
                    *(
                        make_inbetween(
                            InBetweenParams(
                                from_=pair[0],
                                to=pair[1],
                                step=step
                            )
                        ) for step in np.linspace(
                            start=1/steps,
                            stop=1,
                            num=steps-1,
                            endpoint=False
                        )
                    )
                ) for pair in pairwise(lst)
            )
        ),
        lst[-1]
    ]

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def repeat_along_dim_0(t: Tensor, factor: int) -> Tensor:
    """
    Repeats a tensor's contents along its 0th dim `factor` times.

    repeat_along_dim_0(torch.tensor([[0,1]]), 2)
    tensor([[0, 1],
            [0, 1]])
    # shape changes from (1, 2)
    #                 to (2, 2)
    
    repeat_along_dim_0(torch.tensor([[0,1],[2,3]]), 2)
    tensor([[0, 1],
            [2, 3],
            [0, 1],
            [2, 3]])
    # shape changes from (2, 2)
    #                 to (4, 2)
    """
    assert factor >= 1
    if factor == 1:
        return t
    if t.size(dim=0) == 1:
        # prefer expand() whenever we can, since doesn't copy
        return t.expand(factor * t.size(dim=0), *(-1,)*(t.ndim-1))
    return t.repeat((factor, *(1,)*(t.ndim-1)))

def repeat_interleave_along_dim_0(t: Tensor, factors: Iterable[int], factors_tensor: Tensor, output_size: int) -> Tensor:
    """
    repeat_interleave()s a tensor's contents along its 0th dim.

    factors=(2,3)
    factors_tensor = torch.tensor(factors)
    output_size=factors_tensor.sum().item() # 5
    t=torch.tensor([[0,1],[2,3]])
    repeat_interleave_along_dim_0(t=t, factors=factors, factors_tensor=factors_tensor, output_size=output_size)
    tensor([[0, 1],
            [0, 1],
            [2, 3],
            [2, 3],
            [2, 3]])
    """
    factors_len = len(factors)
    assert factors_len >= 1
    if len(factors) == 1:
        # prefer repeat() whenever we can, because MPS doesn't support repeat_interleave()
        return repeat_along_dim_0(t, factors[0])
    if t.device.type != 'mps':
        return t.repeat_interleave(factors_tensor, dim=0, output_size=output_size)
    return torch.cat([repeat_along_dim_0(split, factor) for split, factor in zip(t.split(1, dim=0), factors)])

def cat_self_with_repeat_interleaved(t: Tensor, factors: Iterable[int], factors_tensor: Tensor, output_size: int) -> Tensor:
    """
    Fast-paths for a pattern which in its worst-case looks like:
    t=torch.tensor([[0,1],[2,3]])
    factors=(2,3)
    torch.cat((t, t.repeat_interleave(factors, dim=0)))
    tensor([[0, 1],
            [2, 3],
            [0, 1],
            [0, 1],
            [2, 3],
            [2, 3],
            [2, 3]])

    Fast-path:
      `len(factors) == 1`
      it's just a normal repeat
    t=torch.tensor([[0,1]])
    factors=(2)
    tensor([[0, 1],
            [0, 1],
            [0, 1]])
    
    t=torch.tensor([[0,1],[2,3]])
    factors=(2)
    tensor([[0, 1],
            [2, 3],
            [0, 1],
            [2, 3],
            [0, 1],
            [2, 3]])
    """
    if len(factors) == 1:
        return repeat_along_dim_0(t, factors[0]+1)
    return torch.cat((t, repeat_interleave_along_dim_0(t=t, factors_tensor=factors_tensor, factors=factors, output_size=output_size)))

def sum_along_slices_of_dim_0(t: Tensor, arities: Iterable[int]) -> Tensor:
    """
    Implements fast-path for a pattern which in the worst-case looks like this:
    t=torch.tensor([[1],[2],[3]])
    arities=(2,1)
    torch.cat([torch.sum(split, dim=0, keepdim=True) for split in t.split(arities)])
    tensor([[3],
            [3]])

    Fast-path:
      `len(arities) == 1`
      it's just a normal sum(t, dim=0, keepdim=True)
    t=torch.tensor([[1],[2]])
    arities=(2)
    t.sum(dim=0, keepdim=True)
    tensor([[3]])
    """
    if len(arities) == 1:
        if t.size(dim=0) == 1:
            return t
        return t.sum(dim=0, keepdim=True)
    splits: List[Tensor] = t.split(arities)
    del t
    sums: List[Tensor] = [torch.sum(split, dim=0, keepdim=True) for split in splits]
    del splits
    return torch.cat(sums)


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


# def check_safety(x_image):
#     safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
#     x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
#     assert x_checked_image.shape[0] == len(has_nsfw_concept)
#     for i in range(len(has_nsfw_concept)):
#         if has_nsfw_concept[i]:
#             x_checked_image[i] = load_replacement(x_checked_image[i])
#     return x_checked_image, has_nsfw_concept

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

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

@dataclass
class WeightedPrompt():
    text: str
    weight: float

def parse_prompt(prompt: str) -> WeightedPrompt:
    match = re.search(r"^-?\d+(\.\d*)?:", prompt)
    if match is None:
        return WeightedPrompt(text=prompt, weight=1.0)
    group = match.group()[:-1]
    weight = float(group)
    return WeightedPrompt(text=prompt[len(group)+1:], weight=weight)

MultiPrompt: TypeAlias = Iterable[WeightedPrompt]

@dataclass
class SampleSpec(abc.ABC):
    multiprompt: MultiPrompt

# https://stackoverflow.com/a/73107990/5257399
class InterpStrategy(str, Enum):
    Slerp = 'slerp'
    Lerp = 'lerp'

    def __str__(self) -> str:
        return self.value

@dataclass
class LatentWalkSampleSpec(SampleSpec):
    target_multiprompt: MultiPrompt
    interp_quotient: float
    interp_strategy: InterpStrategy
SampleSpec.register(LatentWalkSampleSpec)

@dataclass
class IdenticalSamplesBatchSpec():
    sample: SampleSpec

@dataclass
class VariedSamplesBatchSpec():
    samples: List[SampleSpec]

@dataclass
class BatchSpec(abc.ABC): pass
BatchSpec.register(IdenticalSamplesBatchSpec)
BatchSpec.register(VariedSamplesBatchSpec)

def main():
    parser = argparse.ArgumentParser()

    # proposed_seed = randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
    proposed_seed = 42

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        action='append',
        default=[],
        help="the prompt to render. you can express your prompt as '0.5:piano villain' to halve its effect. negative numbers accepted. try multiple prompts with differing weights. --scale can be used to further amplify the difference between your summed prompts and the unconditional prompt."
    )
    parser.add_argument(
        "--prompt_interpolation_steps",
        type=int,
        nargs="?",
        default=None,
        help="additionally emit samples of the latent walk between each supplied prompt. specifies the number of steps over which to perform the interpolation (e.g. how many samples to emit -- more steps = more gradual interpolation, more images produced)."
    )
    parser.add_argument(
        "--prompt_interpolation_strategy",
        type=str,
        choices=[InterpStrategy.Slerp, InterpStrategy.Lerp],
        default=InterpStrategy.Slerp,
        help="for latent walks (see --prompt_interpolation_steps): whether to use spherical or lineral interpolation between latent coordinates. spherical interpolation is more correct (see Kai Christensen's explanation in the text description of https://www.youtube.com/watch?v=nkliw2ue5ak)."
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
    # my recommendations for each sampler are:
    # implement samplers from Karras et al paper using Karras noise schedule
    # if your step count is low (for example 7 or 8) you should use add --end_karras_ramp_early too.
    # --heun --karras_noise
    # --euler --karras_noise
    # --dpm2 --karras_noise
    # I assume Karras noise schedule is generally applicable, so is suitable for use with any k-diffusion sampler.
    # --k_lms --karras_noise
    # --euler_ancestral --karras_noise
    # --dpm2_ancestral --karras_noise
    # I didn't implement any way to generate the DDIM/PLMS sigmas from a Karras noise schedule
    # --ddim
    # --plms
    parser.add_argument(
        "--karras_noise",
        action='store_true',
        help=f"use noise schedule from arXiv:2206.00364. Implemented for k-diffusion samplers, {K_DIFF_SAMPLERS}. but you should probably use it with one of the samplers introduced in the same paper: {KARRAS_SAMPLERS}.",
    )
    parser.add_argument(
        "--end_karras_ramp_early",
        action='store_true',
        help=f"when --karras_noise is enabled: ramp from sigma_max (14.6146) to a sigma *slightly above* sigma_min (0.0292), instead of including sigma_min in our ramp. because the effect on the image of sampling sigma_min is not very big, and every sigma counts when our step counts are low. use this to get good results with {KARRAS_SAMPLERS} at step counts as low as 7 or 8.",
    )
    parser.add_argument(
        "--dynamic_thresholding",
        action='store_true',
    )
    parser.add_argument(
        "--dynamic_thresholding_percentile",
        type=float,
        default=0.9995,
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
    parser.add_argument(
        "--filename_prompt",
        action='store_true',
        help="include prompt in filename",
    )
    parser.add_argument(
        "--filename_sample_ix",
        action='store_true',
        help="include (sample-within-batch index, batch index) in file name",
    )
    parser.add_argument(
        "--filename_seed",
        action='store_true',
        help="include seed in file name",
    )
    parser.add_argument(
        "--filename_sampling",
        action='store_true',
        help="include sampling config in file name",
    )
    parser.add_argument(
        "--filename_guidance",
        action='store_true',
        help="include guidance config in file name",
    )
    parser.add_argument(
        "--filename_sigmas",
        action='store_true',
        help="include sigmas in file name",
    )
    parser.add_argument(
        "--init_img",
        type=str,
        nargs="?",
        help="path to the input image"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"


    config = OmegaConf.load(f"{opt.config}")
    model: LatentDiffusion = load_model_from_config(config, f"{opt.ckpt}")

    seed_everything(opt.seed)

    device = torch.device(get_device())
    model = model.to(device)

    # if opt.sampler in K_DIFF_SAMPLERS:
    #     model_k_wrapped = CompVisDenoiser(model, quantize=True)
    #     model_k_guidance = KCFGDenoiser(model_k_wrapped)
    # elif opt.sampler in NOT_K_DIFF_SAMPLERS:
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

    prompts_change_each_batch = bool(opt.from_file)
    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    make_inbetween: MakeInbetween[SampleSpec] = lambda params: LatentWalkSampleSpec(
        interp_quotient=params.step,
        interp_strategy=opt.prompt_interpolation_strategy,
        multiprompt=params.from_.multiprompt,
        target_multiprompt=params.to.multiprompt
    )

    batch_specs: Iterable[BatchSpec] = None
    if opt.from_file:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            lines = f.read().splitlines()
            batch_specs = [
                VariedSamplesBatchSpec(samples=batch) for batch in chunk(
                    intersperse_linspace(
                        [
                            # every line in the file is considered to be a single multiprompt.
                            SampleSpec(
                                # splitting the line on tab, gives each prompt of the multiprompt.
                                multiprompt=[parse_prompt(subprompt) for subprompt in line.split('\t')]
                            ) for line in lines
                        ],
                        make_inbetween=make_inbetween,
                        steps=opt.prompt_interpolation_steps
                    ),
                    batch_size
                )
            ]
    elif len(opt.prompt) == 1:
        # fast-path for the common case where just one prompt is provided
        batch_specs = [IdenticalSamplesBatchSpec(
            sample = SampleSpec(
                multiprompt=[parse_prompt(subprompt) for subprompt in opt.prompt[0]]
            )
        )]
    else:
        batch_specs = [
            VariedSamplesBatchSpec(samples=batch) for batch in chunk(
                intersperse_linspace(
                    [
                        SampleSpec(
                            multiprompt=[parse_prompt(subprompt) for subprompt in multiprompt]
                        ) for multiprompt in opt.prompt
                    ],
                    make_inbetween=make_inbetween,
                    steps=opt.prompt_interpolation_steps
                ),
                batch_size
            )
        ]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    karras_noise_active = False
    end_karras_ramp_early_active = False

    def _format_sigma_pretty(sigma: Tensor) -> str:
        return "%.4f" % sigma

    def format_sigmas_pretty(sigmas: Tensor, summary: bool = False) -> str:
        if (summary and sigmas.size(dim=0) > 9):
            start = ", ".join(_format_sigma_pretty(sigma) for sigma in sigmas[0:4])
            end = ", ".join(_format_sigma_pretty(sigma) for sigma in sigmas[-4:])
            return f'[{start}, …, {end}]'
        return f'[{", ".join(_format_sigma_pretty(sigma) for sigma in sigmas)}]'

    def _compute_common_file_name_portion(sample_ix: str = '', sigmas: str = '') -> str:
        seed = ''
        sampling = ''
        prompt = ''
        sample_ix_ = ''
        sigmas_ = ''
        guidance = ''
        if opt.filename_sampling:
            kna = '_kns' if karras_noise_active else ''
            nz = '_ek' if end_karras_ramp_early_active else ''
            sampling = f"{opt.sampler}{opt.steps}{kna}{nz}"
        if opt.filename_seed:
            seed = f".s{opt.seed}"
        if opt.filename_prompt:
            sanitized = re.sub(r"[/\\?%*:|\"<>\x7F\x00-\x1F]", "-", '_'.join(opt.prompt))
            prompt = f"_{sanitized}_"
        if opt.filename_sample_ix:
            sample_ix_ = sample_ix
        if opt.filename_sigmas and sigmas is not None:
            sigmas_ = f"_{sigmas}_"
        if opt.filename_guidance:
            guidance = f"_str{opt.strength}_sca{opt.scale}"
        nominal = f"{seed}{sample_ix_}{prompt}{sigmas_}{guidance}{sampling}"
        # https://apple.stackexchange.com/a/86617/251820
        # macOS imposes a filename limit of ~255 chars
        # we already used up some on base_count and the file extension
        # shed the biggest parts if we must, so that saving doesn't go bang
        if len(nominal) > 245:
            nominal = f"{seed}{sample_ix_}{prompt}{guidance}{sampling}"
        if len(nominal) > 245:
            nominal = f"{seed}{sample_ix_}{guidance}{sampling}"
        return nominal

    def compute_batch_file_name(sigmas: str = '') -> str:
        common_file_name_portion = _compute_common_file_name_portion(sigmas=sigmas)
        return f"grid-{grid_count:04}{common_file_name_portion}.png"

    def compute_sample_file_name(batch_ix: int, sample_ix_in_batch: int, sigmas: Optional[str] = None) -> str:
        sample_ix=f".n{batch_ix}.i{sample_ix_in_batch}"
        common_file_name_portion = _compute_common_file_name_portion(sample_ix=sample_ix, sigmas=sigmas)
        return f"{base_count:05}{common_file_name_portion}.png"
    
    def img_to_latent(path: str) -> Tensor:
        assert os.path.isfile(path)
        image = load_img(path).to(device)
        image = repeat(image, '1 ... -> b ...', b=batch_size)
        latent: Tensor = model.get_first_stage_encoding(model.encode_first_stage(image))  # move to latent space
        return latent

    init_latent = None
    if opt.init_img:
        init_latent = img_to_latent(opt.init_img)
    t_enc = int((1.0-opt.strength) * opt.steps)

    start_code = None
    # shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    if opt.fixed_code:
        shape = [opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f]
        # rand_size = [opt.n_samples, *shape]
        # https://github.com/CompVis/stable-diffusion/issues/25#issuecomment-1229706811
        # MPS random is not currently deterministic w.r.t seed, so compute randn() on-CPU
        start_code = torch.randn(shape, device='cpu').to(device) if device.type == 'mps' else torch.randn(shape, device=device)
        seed_everything(opt.seed)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    if device.type == 'mps':
        precision_scope = nullcontext # have to use f32 on mps
    with torch.no_grad():
        with precision_scope(device.type):
            with model.ema_scope():
                tic = time.perf_counter()
                all_samples = list()
                c: Optional[FloatTensor] = None
                cond_weights: Optional[Iterable[float]] = None
                cond_arities: Optional[Iterable[int]] = None
                for n in trange(opt.n_iter, desc="Iterations", disable=True):
                    iter_tic = time.perf_counter()
                    for batch_spec in tqdm(batch_specs, desc=f"Iteration {n}, batch", disable=True):
                        uc = None if opt.scale == 1.0 else model.get_learned_conditioning("").expand(batch_size, -1, -1)
                        # if c is None or prompts_change_each_batch:
                        match batch_spec:
                            case IdenticalSamplesBatchSpec(sample):
                                # for some reason Python isn't narrowing the type automatically
                                sample: SampleSpec = sample
                                texts: List[str] = [subprompt.text for subprompt in sample.multiprompt]
                                cond_arities: Tuple[int, ...] = (len(texts),) * batch_size
                                cond_weights: List[float] = [subprompt.weight for subprompt in sample.multiprompt] * batch_size
                                p = model.get_learned_conditioning(texts)
                                # print(f"Iteration {n} p: {hash(p.cpu().detach().numpy().tobytes())}")
                                c = repeat_along_dim_0(p, batch_size)
                                del p
                            case VariedSamplesBatchSpec(samples):
                                # for some reason Python isn't narrowing the type automatically
                                samples: List[SampleSpec] = samples
                                assert len(samples) == batch_size
                                texts: List[str] = [subprompt.text for sample in samples for subprompt in sample.multiprompt]
                                cond_weights: List[float] = [subprompt.weight for sample in samples for subprompt in sample.multiprompt]
                                cond_arities: List[int] = [len(sample.multiprompt) for sample in samples]
                                c = model.get_learned_conditioning(texts)
                            case _:
                                raise TypeError(f"That ({batch_spec}) ain't no BatchSpec I ever heard of")
                        # print(f"Iteration {n} uc: {hash(uc.cpu().detach().numpy().tobytes())}")
                        # print(f"Iteration {n} c: {hash(c.cpu().detach().numpy().tobytes())}")

                        # if init_latent is None and (start_code is None or not opt.fixed_code):
                        # if not opt.fixed_code:
                        #     rand_size = [opt.n_samples, *shape]
                        #     # https://github.com/CompVis/stable-diffusion/issues/25#issuecomment-1229706811
                        #     # MPS random is not currently deterministic w.r.t seed, so compute randn() on-CPU
                        #     start_code = torch.randn(rand_size, device='cpu').to(device) if device.type == 'mps' else torch.randn(rand_size, device=device)
                        
                        
                        # print(f"Iteration {n} start_code: {hash(start_code.cpu().detach().numpy().tobytes())}")

                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        if opt.sampler in NOT_K_DIFF_SAMPLERS:
                            # print(f"Using {opt.sampler} so x_T=start_code")
                            if opt.karras_noise:
                                print(f"[WARN] You have requested --karras_noise, but Karras et al noise schedule is not implemented for {opt.sampler} sampler. Implemented only for {K_DIFF_SAMPLERS}. Using default noise schedule from DDIM.")
                            if init_latent is None:
                                samples, _ = sampler.sample(
                                    S=opt.steps,
                                    conditioning=c,
                                    batch_size=opt.n_samples,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=opt.scale,
                                    unconditional_conditioning=uc,
                                    eta=opt.ddim_eta,
                                    x_T=start_code
                                )
                                # for PLMS and DDIM, sigmas are all 0
                                sigmas = None
                                sigmas_quantized = None
                            else:
                                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                                samples = sampler.decode(
                                    z_enc,
                                    c,
                                    t_enc,
                                    unconditional_guidance_scale=opt.scale,
                                    unconditional_conditioning=uc,
                                )
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
                            if opt.karras_noise:
                                if opt.sampler not in KARRAS_SAMPLERS:
                                    print(f"[WARN] you have enabled --karras_noise, but you are using it with a sampler ({opt.sampler}) outside of the ones proposed in the same paper (arXiv:2206.00364), {KARRAS_SAMPLERS}. No idea what results you'll get.")
                                
                                # the idea of "ending the Karras ramp early" (i.e. setting a high sigma_min) is that sigmas as lower as sigma_min
                                # aren't very impactful, and every sigma counts when our step count is low
                                # https://github.com/crowsonkb/k-diffusion/pull/23#issuecomment-1234872495
                                # this is just a more performant way to get the "sigma before sigma_min" from a Karras schedule, aka
                                # get_sigmas_karras(n=steps, sigma_max=sigma_max, sigma_min=sigma_min_nominal, rho=rho)[-3]
                                def get_premature_sigma_min(
                                    steps: int,
                                    sigma_max: float,
                                    sigma_min_nominal: float,
                                    rho: float
                                ) -> float:
                                    min_inv_rho = sigma_min_nominal ** (1 / rho)
                                    max_inv_rho = sigma_max ** (1 / rho)
                                    ramp = (steps-2) * 1/(steps-1)
                                    sigma_min = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
                                    return sigma_min

                                rho = 7.
                                # 14.6146
                                sigma_max=model_k_wrapped.sigmas[-1].item()
                                # 0.0292
                                sigma_min_nominal=model_k_wrapped.sigmas[0].item()
                                # get the "sigma before sigma_min" from a slightly longer ramp
                                # https://github.com/crowsonkb/k-diffusion/pull/23#issuecomment-1234872495
                                premature_sigma_min = get_premature_sigma_min(
                                    steps=opt.steps+1,
                                    sigma_max=sigma_max,
                                    sigma_min_nominal=sigma_min_nominal,
                                    rho=rho
                                )
                                sigmas = get_sigmas_karras(
                                    n=opt.steps,
                                    sigma_min=premature_sigma_min if opt.end_karras_ramp_early else sigma_min_nominal,
                                    sigma_max=sigma_max,
                                    rho=rho,
                                    device=device,
                                )
                                karras_noise_active = True
                                end_karras_ramp_early_active = opt.end_karras_ramp_early
                            else:
                                if opt.sampler in KARRAS_SAMPLERS:
                                    print(f"[WARN] you should really enable --karras_noise for best results; it's the noise schedule proposed in the same paper (arXiv:2206.00364) as the sampler you're using ({opt.sampler}). Falling back to default k-diffusion get_sigmas() noise schedule.")
                                sigmas = model_k_wrapped.get_sigmas(opt.steps)
                            
                            if init_latent is not None:
                                sigmas = sigmas[len(sigmas) - t_enc - 1 :]
                            
                            print('sigmas (before quantization):')
                            print(format_sigmas_pretty(sigmas))
                            print('sigmas (after quantization):')
                            sigmas_quantized = append_zero(model_k_wrapped.sigmas[torch.argmin((sigmas[:-1].reshape(len(sigmas)-1, 1).repeat(1, len(model_k_wrapped.sigmas)) - model_k_wrapped.sigmas).abs(), dim=1)])
                            print(format_sigmas_pretty(sigmas_quantized))

                            x = start_code * sigmas[0] # for GPU draw
                            if init_latent is not None:
                                x = init_latent + x
                            print(f"Iteration {n} x: {hash(x.cpu().detach().numpy().tobytes())}")
                            extra_args = {
                                'cond': c,
                                'uncond': uc,
                                'cond_weights': cond_weights,
                                'cond_arities': cond_arities,
                                'cond_scale': opt.scale,
                            }
                            samples = sampling_fn(
                                model_k_guidance,
                                x,
                                sigmas,
                                extra_args=extra_args,
                                **noise_schedule_sampler_args)

                        x_samples = model.decode_first_stage(samples)

                        if opt.dynamic_thresholding:
                            # https://github.com/lucidrains/imagen-pytorch/blob/ceb23d62ecf611082c82b94f2625d78084738ced/imagen_pytorch/imagen_pytorch.py#L1982
                            # adapted from lucidrains' imagen_pytorch (MIT-licensed)
                            flattened = rearrange(x_samples, 'a b ... -> a b (...)').abs()
                            # aten::sort.values_stable not implemented for MPS
                            sort_on_cpu = device.type == 'mps'
                            flattened = flattened.cpu() if sort_on_cpu else flattened
                            # implementation of pseudocode from Imagen paper https://arxiv.org/abs/2205.11487 Section E, A.32
                            s = torch.quantile(
                                flattened,
                                opt.dynamic_thresholding_percentile,
                                dim = 2
                            )
                            s = s.to(device) if sort_on_cpu else s
                            s.clamp_(min = 1.)
                            s = right_pad_dims_to(x_samples, s)
                            # MPS complains min and input tensors must be of the same shape

                            clamp_tensors_on_cpu = device.type == 'mps'
                            s_orig = s
                            neg_s = -s
                            s = s.cpu() if clamp_tensors_on_cpu else s
                            neg_s = neg_s.cpu() if clamp_tensors_on_cpu else neg_s
                            x_samples = x_samples.cpu() if clamp_tensors_on_cpu else x_samples
                            x_samples = x_samples.clamp(neg_s, s)
                            x_samples = x_samples.to(device) if clamp_tensors_on_cpu else x_samples
                            x_samples = x_samples / s_orig
                        
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image, has_nsfw_concept = check_safety_poorly(x_samples)

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        if not opt.skip_save:
                            for ix, x_sample in enumerate(x_checked_image_torch):
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                # img = put_watermark(img, wm_encoder)
                                preferred_sigmas = sigmas_quantized if sigmas_quantized is not None else sigmas
                                img.save(os.path.join(sample_path, compute_sample_file_name(batch_ix=n, sample_ix_in_batch=ix, sigmas=format_sigmas_pretty(preferred_sigmas, summary=True) if preferred_sigmas is not None else None)))
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
                    img.save(os.path.join(outpath, compute_batch_file_name()))
                    grid_count += 1

                toc = time.perf_counter()
                print(f'in total, generated {opt.n_iter} batches of {batch_size} images in {toc-tic} seconds')

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
