# from torch import save, load, tensor
# from torch.nn import ParameterDict
from ldm.modules.embedding_manager import EmbeddingManager

def main():
    embedding_manager = EmbeddingManager()

    # filename = 'test.pt'
    # save({
    #     'string_to_token': {
    #         '*': tensor(265)
    #     },
    #     'string_to_param': ParameterDict()
    # }, filename)
    # x = load(filename, map_location='cpu')

    embedding_manager.load('/Users/birch/git/stable-diffusion/logs/2022-09-20T01-49-11_fumo/checkpoints/embeddings.pt')
    # embedding_manager.load(filename)

    embedding_manager.repro()


if __name__ == '__main__':
    main()
