"""SAMPLING ONLY."""

import torch

from .dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver

# this file comes from Cheng Lu's dpm-solver repository, MIT-licensed.
# https://github.com/LuChengTHU/dpm-solver/blob/414c74f62fb189723461aadc91dc6527301e1dbe/example_v2/stable-diffusion/ldm/models/diffusion/dpm_solver/sampler.py
# https://github.com/LuChengTHU/dpm-solver/blob/414c74f62fb189723461aadc91dc6527301e1dbe/LICENSE

class DPMSolverSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        to_torch = lambda x: x.detach().clone().float().to(model.device)
        self.register_buffer('alphas_cumprod', to_torch(model.alphas_cumprod))

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
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

        dpm_solver = DPM_Solver(model_fn, ns, predict_x0=True, thresholding=False)
        x = dpm_solver.sample(img, steps=S, skip_type="time_uniform", method="multistep", order=2, lower_order_final=True)

        return x.to(device), None