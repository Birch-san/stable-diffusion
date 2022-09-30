import torch
from transformers.models.clip.modeling_clip import CLIPTextEmbeddings
from transformers.models.clip.configuration_clip import CLIPTextConfig

from ldm.modules.embedding_manager import EmbeddingManager

class FrozenCLIPEmbedder():
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14"):
        config = CLIPTextConfig.from_pretrained(version)
        self.embeddings = CLIPTextEmbeddings(config).to('mps')

        def embedding_forward(
            self,
            input_ids = None,
            position_ids = None,
            inputs_embeds = None,
            embedding_manager = None,
        ) -> torch.Tensor:
            position_ids = torch.arange(0, 77, device='mps').unsqueeze(0)

            if inputs_embeds is None:
                inputs_embeds = self.token_embedding(input_ids)

            if embedding_manager is not None:
                inputs_embeds = embedding_manager(input_ids, inputs_embeds)

            self.position_embedding(position_ids)      

        self.embeddings.forward = embedding_forward.__get__(self.embeddings)

    def repro(self, embedding_manager: EmbeddingManager):
        self.embeddings(input_ids=torch.cat([torch.tensor([49406], device='mps'), torch.tensor([49407], device='mps').expand(76)]).unsqueeze(0), embedding_manager=embedding_manager)
