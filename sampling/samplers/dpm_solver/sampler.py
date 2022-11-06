"""SAMPLING ONLY."""

import torch

from .dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
from util.device import get_device_type

# this file comes from Cheng Lu's dpm-solver repository, MIT-licensed.
# https://github.com/LuChengTHU/dpm-solver/blob/414c74f62fb189723461aadc91dc6527301e1dbe/example_v2/stable-diffusion/ldm/models/diffusion/dpm_solver/sampler.py
# https://github.com/LuChengTHU/dpm-solver/blob/414c74f62fb189723461aadc91dc6527301e1dbe/LICENSE

class DPMSolverSampler(object):
    predict_x0: bool
    def __init__(
        self,
        model,
        predict_x0=False,
        **kwargs
    ):
        """
        We support both the noise prediction model ("predicting epsilon") and the data prediction model ("predicting x0").
        If `predict_x0` is False, we use the solver for the noise prediction model (DPM-Solver).
        If `predict_x0` is True, we use the solver for the data prediction model (DPM-Solver++).
            In such case, we further support the "dynamic thresholding" in [1] when `thresholding` is True.
            The "dynamic thresholding" can greatly improve the sample quality for pixel-space DPMs with large guidance scales.
        
        Args:
            model
            predict_x0: A `bool`. If true, use the data prediction model; else, use the noise prediction model.
        """
        super().__init__()
        self.model = model
        self.predict_x0=predict_x0
        to_torch = lambda x: x.detach().clone().float().to(model.device)
        preferred_device = torch.device(get_device_type())
        self.register_buffer('alphas_cumprod', to_torch(model.alphas_cumprod), preferred_device)

    def register_buffer(self, name, attr, preferred_device: torch.device):
        if type(attr) == torch.Tensor:
            if attr.device.type != preferred_device.type:
                attr = attr.to(preferred_device)
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        # print(f'Data shape for DPM-Solver sampling is {size}, sampling steps {S}')

        device = self.model.betas.device
        if x_T is None:
            # https://github.com/CompVis/stable-diffusion/issues/25#issuecomment-1229706811
            # MPS random is not currently deterministic w.r.t seed, so compute randn() on-CPU
            img = torch.randn(size, device='cpu' if device.type == 'mps' else device).to(device)
        else:
            img = x_T

        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)

        model_fn = model_wrapper(
            lambda x, t, c: self.model.apply_model(x, t, c),
            ns,
            model_type="noise",
            guidance_type="classifier-free",
            condition=conditioning,
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )

        dpm_solver = DPM_Solver(model_fn, ns, predict_x0=self.predict_x0)
        x = dpm_solver.sample(img, steps=S, skip_type="time_uniform", method="multistep", order=2, lower_order_final=True)

        return x.to(device), None