from torch.nn import ParameterDict, Parameter
from torch import tensor, rand, load, cat, save, as_tensor, float32
from torch.nn import Embedding

def main():
    device='mps'
    token_embedding = Embedding(49408, 768, device=device)
    
    # filename = 'test.pt'
    # save({
    #     'string_to_token': {
    #         '*': tensor(265)
    #     },
    #     'string_to_param': ParameterDict(
    #         parameters={
    #             '*': Parameter(
    #                     as_tensor(rand(
    #                         size=(1, 768),
    #                         requires_grad=True,
    #                         dtype=float32,
    #                         device='mps'
    #                     ), device='mps')
    #                 )
    #         }
    #     )
    # }, filename)

    # ckpt = load(filename, map_location='cpu')
    ckpt = load('embeddings.pt', map_location='cpu')

    # Handle .pt textual inversion files
    # self.string_to_token_dict = { '*': tensor(265, device='cpu') }
    assert 'string_to_token' in ckpt
    string_to_token_dict = ckpt["string_to_token"]

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
