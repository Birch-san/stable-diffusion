import torch
from torch.nn import Embedding

class EmbeddingManager():
    def __init__(self):
        self.token_embedding = Embedding(49408, 768, device='mps')

        self.string_to_token_dict = {}

        token = torch.tensor(265, device='mps')

        self.string_to_token_dict['*'] = token

    def repro(self):
        tokenized_text = torch.cat([torch.tensor([49406], device='mps'), torch.tensor([49407], device='mps').expand(76)]).unsqueeze(0)
        self.token_embedding(tokenized_text)
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
