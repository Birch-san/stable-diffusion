import torch
import torch.nn as nn
from transformers import CLIPTextModel

class FrozenCLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device='mps'):
        super().__init__()
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.freeze()

        def embedding_forward(
            self,
            input_ids = None,
            position_ids = None,
            inputs_embeds = None,
            embedding_manager = None,
        ) -> torch.Tensor:

            seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

            if position_ids is None:
                position_ids = self.position_ids[:, :seq_length]

            if inputs_embeds is None:
                inputs_embeds = self.token_embedding(input_ids)

            if embedding_manager is not None:
                inputs_embeds = embedding_manager(input_ids, inputs_embeds)


            position_embeddings = self.position_embedding(position_ids)
            embeddings = inputs_embeds + position_embeddings

            return embeddings      

        self.transformer.text_model.embeddings.forward = embedding_forward.__get__(self.transformer.text_model.embeddings)


        def text_encoder_forward(
            self,
            input_ids = None,
            position_ids = None,
            embedding_manager = None,
        ):
            self.embeddings(input_ids=input_ids, position_ids=position_ids, embedding_manager=embedding_manager)

        self.transformer.text_model.forward = text_encoder_forward.__get__(self.transformer.text_model)

        def transformer_forward(
            self,
            input_ids = None,
            position_ids = None,
            embedding_manager = None,
        ):
            return self.text_model(
                input_ids=input_ids,
                position_ids=position_ids,
                embedding_manager = embedding_manager
            )

        self.transformer.forward = transformer_forward.__get__(self.transformer)

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def repro(self, **kwargs):
        self.transformer(input_ids=torch.cat([torch.tensor([49406], device='mps'), torch.tensor([49407], device='mps').expand(76)]).unsqueeze(0), **kwargs)
