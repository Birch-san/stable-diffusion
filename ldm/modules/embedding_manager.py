import torch
from torch import nn, tensor

from ldm.data.personalized import per_img_token_list
from functools import partial

def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))[0, 0]


class EmbeddingManager(nn.Module):
    def __init__(
        self,
        embedder,
        placeholder_strings=None,
        initializer_words=None,
        per_image_tokens=False,
        num_vectors_per_token=1,
        progressive_words=False,
        **kwargs,
    ):
        super().__init__()

        self.embedder = embedder

        self.string_to_token_dict = {}

        if per_image_tokens:
            placeholder_strings.extend(per_img_token_list)

        for idx, placeholder_string in enumerate(placeholder_strings):

            token = torch.tensor(265, device=embedder.device)

            self.string_to_token_dict[placeholder_string] = token

    def forward(
        self,
        tokenized_text,
        embedded_text,
    ):
        placeholder_token = self.string_to_token_dict['*']

        cpu_item = placeholder_token.item()
        assert cpu_item == 265
        # placeholder_token = placeholder_token.detach().clone().to(tokenized_text.device)
        placeholder_token = placeholder_token.to(tokenized_text.device)
        gpu_item = placeholder_token.item()
        assert gpu_item == cpu_item, f"GPU item was: {gpu_item}, expected {cpu_item}. This indicates failure to transfer tensor from CPU to GPU"

        assert False, "Okay, if you got this far then there's no problem. I simplified this function too much to return the right thing though, so let's abort."

    def load(self, ckpt_path, full=True):
        ckpt = torch.load(ckpt_path, map_location='cpu')

        # Handle .pt textual inversion files
        # self.string_to_token_dict = { '*': tensor(265, device='cpu') }
        if 'string_to_token' in ckpt:
            self.string_to_token_dict = ckpt["string_to_token"]
