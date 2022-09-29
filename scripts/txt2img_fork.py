from omegaconf import OmegaConf
from ldm.models.diffusion.ddpm import CondStage

from ldm.util import instantiate_from_config

def main():
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
    modelCS: CondStage = instantiate_from_config(config.modelCondStage)
    modelCS.to('mps')
    modelCS.eval()

    modelCS.embedding_manager.load("/Users/birch/git/stable-diffusion/logs/2022-09-20T01-49-11_fumo/checkpoints/embeddings.pt")

    modelCS.get_learned_conditioning("")


if __name__ == "__main__":
    main()
