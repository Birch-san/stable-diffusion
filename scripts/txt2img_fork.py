import torch
from omegaconf import OmegaConf
from ldm.models.diffusion.ddpm import CondStage

from ldm.util import instantiate_from_config

def load_model_from_config(ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd

def main():
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")

    sd = load_model_from_config("models/ldm/stable-diffusion-v1/model.ckpt")
    li, lo = [], []
    for key, _ in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    modelCS: CondStage = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.to('mps')
    modelCS.eval()

    modelCS.embedding_manager.load("/Users/birch/git/stable-diffusion/logs/2022-09-20T01-49-11_fumo/checkpoints/embeddings.pt")

    modelCS.get_learned_conditioning("")


if __name__ == "__main__":
    main()
