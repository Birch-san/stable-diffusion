from cmath import log
import torch
from torch import nn

import sys

from ldm.data.personalized import per_img_token_list
from transformers import CLIPTokenizer
from functools import partial

DEFAULT_PLACEHOLDER_TOKEN = ['*']

PROGRESSIVE_SCALE = 2000


def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(
        string,
        truncation=True,
        max_length=77,
        return_length=True,
        return_overflowing_tokens=False,
        padding='max_length',
        return_tensors='pt',
    )
    tokens = batch_encoding['input_ids']
    """ assert (
        torch.count_nonzero(tokens - 49407) == 2
    ), f"String '{string}' maps to more than a single token. Please use another string" """

    return tokens[0, 1]


def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    # assert torch.count_nonzero(token) == 3, f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0, 1]

    return token


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

        if hasattr(
            embedder, 'tokenizer'
        ):   # using Stable Diffusion's CLIP encoder
            self.is_clip = True
            get_token_for_string = partial(
                get_clip_token_for_string, embedder.tokenizer
            )
            get_embedding_for_tkn = partial(
                get_embedding_for_clip_token,
                embedder.transformer.text_model.embeddings,
            )
            token_dim = 1280
        else:   # using LDM's BERT encoder
            self.is_clip = False
            get_token_for_string = partial(
                get_bert_token_for_string, embedder.tknz_fn
            )
            get_embedding_for_tkn = embedder.transformer.token_emb
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

    def save(self, ckpt_path):
        torch.save(
            {
                'string_to_token': self.string_to_token_dict,
                'string_to_param': self.string_to_param_dict,
            },
            ckpt_path,
        )

    def load(self, ckpt_path, full=True):
        ckpt = torch.load(ckpt_path, map_location='cpu')

        # Handle .pt textual inversion files
        if 'string_to_token' in ckpt and 'string_to_param' in ckpt:
            self.string_to_token_dict = ckpt["string_to_token"]
            self.string_to_param_dict = ckpt["string_to_param"]

        # Handle .bin textual inversion files from Huggingface Concepts
        # https://huggingface.co/sd-concepts-library
        else:
            for token_str in list(ckpt.keys()):
                token = get_clip_token_for_string(self.embedder.tokenizer, token_str)
                self.string_to_token_dict[token_str] = token
                ckpt[token_str] = torch.nn.Parameter(ckpt[token_str])
                
            self.string_to_param_dict.update(ckpt)

        if not full:
            for key, value in self.string_to_param_dict.items():
                self.string_to_param_dict[key] = torch.nn.Parameter(value.half())

        print(f'Added terms: {", ".join(self.string_to_param_dict.keys())}')

    def get_embedding_norms_squared(self):
        all_params = torch.cat(
            list(self.string_to_param_dict.values()), axis=0
        )   # num_placeholders x embedding_dim
        param_norm_squared = (all_params * all_params).sum(
            axis=-1
        )              # num_placeholders

        return param_norm_squared

    def embedding_parameters(self):
        return self.string_to_param_dict.parameters()

    def embedding_to_coarse_loss(self):

        loss = 0.0
        num_embeddings = len(self.initial_embeddings)

        for key in self.initial_embeddings:
            optimized = self.string_to_param_dict[key]
            coarse = self.initial_embeddings[key].clone().to(optimized.device)

            loss = (
                loss
                + (optimized - coarse)
                @ (optimized - coarse).T
                / num_embeddings
            )

        return loss
