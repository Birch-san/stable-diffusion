# from torch import save, load, tensor
# from torch.nn import ParameterDict
from torch import tensor, load, cat
from torch.nn import Embedding
from ldm.modules.embedding_manager import EmbeddingManager

def main():
    device='mps'
    token_embedding = Embedding(49408, 768, device=device)

    string_to_token_dict = {}

    token = tensor(265, device=device)

    string_to_token_dict['*'] = token
    
    ckpt = load('/Users/birch/git/stable-diffusion/logs/2022-09-20T01-49-11_fumo/checkpoints/embeddings.pt', map_location='cpu')

    # Handle .pt textual inversion files
    # self.string_to_token_dict = { '*': tensor(265, device='cpu') }
    if 'string_to_token' in ckpt:
        string_to_token_dict = ckpt["string_to_token"]

    # filename = 'test.pt'
    # save({
    #     'string_to_token': {
    #         '*': tensor(265)
    #     },
    #     'string_to_param': ParameterDict()
    # }, filename)
    # x = load(filename, map_location='cpu')

    # embedding_manager.load('/Users/birch/git/stable-diffusion/logs/2022-09-20T01-49-11_fumo/checkpoints/embeddings.pt')
    # embedding_manager.load(filename)

    # embedding_manager.repro()

    tokenized_text = cat([tensor([49406], device=device), tensor([49407], device=device).expand(76)]).unsqueeze(0)
    token_embedding(tokenized_text)
    placeholder_token = string_to_token_dict['*']

    cpu_item = placeholder_token.item()
    assert cpu_item == 265
    # placeholder_token = placeholder_token.detach().clone().to(tokenized_text.device)
    placeholder_token = placeholder_token.to(tokenized_text.device)
    gpu_item = placeholder_token.item()
    assert gpu_item == cpu_item, f"GPU item was: {gpu_item}, expected {cpu_item}. This indicates failure to transfer tensor from CPU to GPU"

    print("Okay, if you got this far then there's no problem.")


if __name__ == '__main__':
    main()
