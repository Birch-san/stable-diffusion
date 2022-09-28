import torch
from omegaconf import OmegaConf
from ldm.models.diffusion.ddpm import LatentDiffusion

from ldm.util import instantiate_from_config


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

    model.to('mps')
    model.eval()
    return model

def main():
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
    model: LatentDiffusion = load_model_from_config(config, "models/ldm/stable-diffusion-v1/model.ckpt")
    model.embedding_manager.load("/Users/birch/git/stable-diffusion/logs/2022-09-20T01-49-11_fumo/checkpoints/embeddings.pt")

    model.get_learned_conditioning("")


if __name__ == "__main__":
    main()
