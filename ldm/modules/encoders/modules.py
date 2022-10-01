import torch
from torch.nn import Embedding

from ldm.modules.embedding_manager import EmbeddingManager

class FrozenCLIPEmbedder():
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self):
        self.token_embedding = Embedding(49408, 768, device='mps')
        self.position_embedding = Embedding(77, 768, device='mps')

    def repro(
        self,
        embedding_manager: EmbeddingManager,
    ) -> torch.Tensor:
        input_ids=torch.cat([torch.tensor([49406], device='mps'), torch.tensor([49407], device='mps').expand(76)]).unsqueeze(0)
        position_ids = torch.arange(0, 77, device='mps').unsqueeze(0)

        inputs_embeds = self.token_embedding(input_ids)
        inputs_embeds = embedding_manager(input_ids, inputs_embeds)

        self.position_embedding(position_ids)