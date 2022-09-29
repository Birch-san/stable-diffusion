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
        self.string_to_param_dict = nn.ParameterDict()

        self.initial_embeddings = (
            nn.ParameterDict()
        )   # These should not be optimized

        self.progressive_words = progressive_words
        self.progressive_counter = 0

        self.max_vectors_per_token = num_vectors_per_token

        self.is_clip = True
        get_token_for_string = lambda _: torch.tensor(265, device=embedder.device)
        get_embedding_for_tkn = partial(
            get_embedding_for_clip_token,
            embedder.transformer.text_model.embeddings,
        )
        token_dim = 1280

        if per_image_tokens:
            placeholder_strings.extend(per_img_token_list)

        for idx, placeholder_string in enumerate(placeholder_strings):

            token = get_token_for_string(placeholder_string)

            if initializer_words and idx < len(initializer_words):
                init_word_token = get_token_for_string(initializer_words[idx])

                with torch.no_grad():
                    init_word_embedding = get_embedding_for_tkn(
                        init_word_token.cpu()
                    )

                token_params = torch.nn.Parameter(
                    init_word_embedding.unsqueeze(0).repeat(
                        num_vectors_per_token, 1
                    ),
                    requires_grad=True,
                )
                self.initial_embeddings[
                    placeholder_string
                ] = torch.nn.Parameter(
                    init_word_embedding.unsqueeze(0).repeat(
                        num_vectors_per_token, 1
                    ),
                    requires_grad=False,
                )
            else:
                token_params = torch.nn.Parameter(
                    torch.rand(
                        size=(num_vectors_per_token, token_dim),
                        requires_grad=True,
                    )
                )

            self.string_to_token_dict[placeholder_string] = token
            self.string_to_param_dict[placeholder_string] = token_params

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
        if 'string_to_token' in ckpt and 'string_to_param' in ckpt:
            self.string_to_token_dict = ckpt["string_to_token"]
            # self.string_to_param_dict = ckpt["string_to_param"]

        print(f'Added terms: {", ".join(self.string_to_param_dict.keys())}')
